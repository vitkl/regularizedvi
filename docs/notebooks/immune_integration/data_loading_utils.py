"""Per-dataset loader functions for immune integration.

Each loader returns a standardized anndata with:
- .X = raw counts (sparse)
- .layers["counts"] = raw counts
- var_names = ENSEMBL IDs, var["SYMBOL"] = gene symbols
- Standardized obs columns (see STANDARD_OBS_COLS)
"""

import gc
import os
import re
import subprocess
from pathlib import Path

import anndata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STANDARD_OBS_COLS = [
    "batch",
    "site",
    "donor",
    "dataset",
    "tissue",
    "condition",
    "age_group",
    "sex",
    "original_annotation",
    "harmonized_annotation",
    "fragment_file_path",
]

# Harmonization mappings (original_label -> harmonized_name) per dataset.
# Derived from annotation_harmonization.md.

BONE_MARROW_MAP = {
    "CD8+ T activated": "CD8+ T",
    "CD14+ Mono": "CD14+ Mono",
    "NK": "NK",
    "CD4+ T activated": "CD4+ T activated",
    "Naive CD20+ B": "Naive B",
    "Erythroblast": "Erythroblast",
    "CD4+ T naive": "CD4+ T naive",
    "Transitional B": "Transitional B",
    "Proerythroblast": "Proerythroblast",
    "CD16+ Mono": "CD16+ Mono",
    "B1 B": "B1 B",
    "Normoblast": "Normoblast",
    "Early Lymphoid": "Lymph prog",
    "G/M prog": "G/M prog",
    "pDC": "pDC",
    "HSC": "HSC",
    "CD8+ T naive": "CD8+ T naive",
    "MK/E prog": "MK/E prog",
    "cDC2": "cDC2",
    "ILC": "ILC",
    "Plasma": "Plasma cell",
    "Other Myeloid": "ID2-hi myeloid prog",
}

TEA_SEQ_MAP = {
    "CD4 Naive": "CD4+ T naive",
    "CD4 TCM": "CD4+ T central memory",
    "B naive": "Naive B",
    "CD14 Mono": "CD14+ Mono",
    "CD8 TEM": "CD8+ T effector memory",
    "CD8 Naive": "CD8+ T naive",
    "NK": "NK",
    "CD4 TEM": "CD4+ T effector memory",
    "B intermediate": "B intermediate",
    "MAIT": "MAIT",
    "Treg": "Treg",
    "CD16 Mono": "CD16+ Mono",
    "B memory": "B memory",
    "gdT": "gamma-delta T",
    "CD8 TCM": "CD8+ T central memory",
    "NK_CD56bright": "NK CD56bright",
    "HSPC": "HSC",
    "cDC2": "cDC2",
    "NK Proliferating": "NK proliferating",
    "dnT": "double-negative T",
    "Platelet": "Platelet",
    "ILC": "ILC",
    "ASDC": "ASDC",
    "Plasmablast": "Plasma cell",
    "CD4 CTL": "CD4+ T CTL",
    "CD8 Proliferating": "CD8+ T proliferating",
}

NEAT_SEQ_MAP = {
    "C1": "CD4+ T recently activated",
    "C2": "Treg",
    "C3": "Th17",
    "C4": "CD4+ T central memory",
    "C5": "Th2",
    "C6": "Th1",
    "C7": "CD4+ T uncommitted memory",
}

CROHNS_MAP = {
    "CD14+ Monocytes": "CD14+ Mono",
    "Tcm": "T central memory",
    "Naive CD4+ T Cells": "CD4+ T naive",
    "NK Cells": "NK",
    "CD8+ Cytotoxic T Cells": "CD8+ T",
    "Transitional B Cells": "Transitional B",
    "FCGR3A+ Monocytes": "CD16+ Mono",
    "Resting Naive B Cells": "Naive B",
    "MAIT Cells": "MAIT",
    "Th1/Th17 Cells": "Th1/Th17",
    "GdT Cells": "gamma-delta T",
    "Activated B Cells": "Activated B",
    "IFN Responding T Cells": "IFN-responding T",
    "Conventional Dendritic Cells": "cDC",
    "Proinflammatory Monocytes": "Proinflammatory Mono",
    "Plasmacytoid Dendritic Cells": "pDC",
    "Plasma B Cells": "Plasma cell",
    "TGFB1+ NK Cells": "NK TGFB1+",
}

LUNG_SPLEEN_MAP = {
    "Memory_B": "B memory",
    "CD8_T": "CD8+ T",
    "NK": "NK",
    "CD4_T": "CD4+ T",
    "Naive_B": "Naive B",
    "Th17": "Th17",
    "Other": "Other",
    "Treg": "Treg",
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _standardize_obs(adata: sc.AnnData) -> sc.AnnData:
    """Ensure all STANDARD_OBS_COLS exist and are string type (NaN stays NaN)."""
    for col in STANDARD_OBS_COLS:
        if col not in adata.obs.columns:
            adata.obs[col] = np.nan
    # Keep only standard columns (drop dataset-specific extras)
    adata.obs = adata.obs[STANDARD_OBS_COLS].copy()
    return adata


def _set_ensembl_var_names(adata: sc.AnnData) -> sc.AnnData:
    """Set ENSEMBL IDs as var_names and gene symbols in var['SYMBOL'].

    Works for 10x H5 files (gene_ids column) and custom h5ad files.
    """
    if "gene_ids" in adata.var.columns:
        adata.var["SYMBOL"] = adata.var_names.values.copy()
        adata.var_names = adata.var["gene_ids"].values.copy()
        adata.var.drop(columns=["gene_ids"], inplace=True)
    elif "ENSEMBL" in adata.var.columns:
        adata.var["SYMBOL"] = adata.var_names.values.copy()
        adata.var_names = adata.var["ENSEMBL"].values.copy()
        adata.var.drop(columns=["ENSEMBL"], inplace=True)
    adata.var_names_make_unique()
    return adata


def _read_10x_mtx(barcodes_path, features_path, matrix_path, sample_id):
    """Read 10x MTX files (barcodes + features + matrix) into AnnData.

    Filters to Gene Expression features only. Sets ENSEMBL as var_names.
    """
    mat = scipy.io.mmread(matrix_path).T.tocsr()
    barcodes = pd.read_csv(barcodes_path, header=None, sep="\t")[0].values
    features = pd.read_csv(features_path, header=None, sep="\t")

    adata = sc.AnnData(
        X=mat,
        obs=pd.DataFrame(index=barcodes),
        var=pd.DataFrame(
            index=features[0].values,  # ENSEMBL IDs as index
        ),
    )
    # features TSV: col0=gene_id, col1=gene_name, col2=feature_type
    adata.var["SYMBOL"] = features[1].values
    if features.shape[1] >= 3:
        adata.var["feature_type"] = features[2].values
        # Filter to Gene Expression
        gex_mask = adata.var["feature_type"] == "Gene Expression"
        if gex_mask.any() and not gex_mask.all():
            adata = adata[:, gex_mask].copy()
        adata.var.drop(columns=["feature_type"], inplace=True)

    adata.var_names_make_unique()

    # Prefix barcodes with sample_id
    adata.obs_names = [f"{sample_id}#{bc}" for bc in adata.obs_names]
    adata.obs["sample"] = sample_id

    return adata


def _finalize(adata: sc.AnnData) -> sc.AnnData:
    """Final steps: ensure uint64 counts, sparse X, standardize obs."""
    if scipy.sparse.issparse(adata.X):
        adata.X = adata.X.tocsr().astype(np.uint16)
    else:
        adata.X = scipy.sparse.csr_matrix(adata.X, dtype=np.uint16)
    adata.layers["counts"] = adata.X.copy()
    adata = _standardize_obs(adata)
    return adata


# ---------------------------------------------------------------------------
# Standalone utilities (avoid regularizedvi dependency)
# ---------------------------------------------------------------------------


_BM_S3_BASE = "s3://openproblems-bio/public/post_competition/multiome"
_BM_BATCHES = [
    "s1d1",
    "s1d2",
    "s1d3",
    "s2d1",
    "s2d4",
    "s2d5",
    "s3d3",
    "s3d6",
    "s3d7",
    "s3d10",
    "s4d1",
    "s4d8",
    "s4d9",
]


def download_bone_marrow_dataset(data_folder: str = "data/") -> str:
    """Download the NeurIPS 2021 bone marrow multiome dataset.

    Downloads the curated h5ad file and per-batch ATAC fragment files
    (+ tabix indices) from the Open Problems S3 bucket if not already
    present locally.

    Returns
    -------
    Path to the downloaded h5ad file.
    """
    os.makedirs(data_folder, exist_ok=True)

    # 1. Download h5ad
    h5ad_name = "bmmc_multiome_multivi_neurips21_curated.h5ad"
    h5ad_path = os.path.join(data_folder, h5ad_name)
    if not os.path.exists(h5ad_path):
        s3_uri = f"{_BM_S3_BASE}/{h5ad_name}"
        print(f"Downloading {h5ad_name} from Open Problems S3 bucket...")
        subprocess.run(
            ["aws", "s3", "cp", s3_uri, h5ad_path, "--no-sign-request"],
            check=True,
        )
        print("Download complete.")
    else:
        print(f"Found {h5ad_path}")

    # 2. Download per-batch fragment files + tabix indices
    for batch in _BM_BATCHES:
        batch_dir = os.path.join(data_folder, batch)
        os.makedirs(batch_dir, exist_ok=True)
        for fname in ["atac_fragments.tsv.gz", "atac_fragments.tsv.gz.tbi"]:
            local_path = os.path.join(batch_dir, fname)
            if not os.path.exists(local_path):
                s3_uri = f"{_BM_S3_BASE}/{batch}/{fname}"
                print(f"Downloading {batch}/{fname}...")
                subprocess.run(
                    ["aws", "s3", "cp", s3_uri, local_path, "--no-sign-request"],
                    check=True,
                )
            else:
                print(f"Found {local_path}")

    return h5ad_path


def _compute_gene_stats(X, batch_size=5000):
    """Compute per-gene n_cells and sum in batches to limit memory."""
    n_cells_total = np.zeros(X.shape[1], dtype=np.int64)
    gene_sum_total = np.zeros(X.shape[1], dtype=np.float64)
    for start in range(0, X.shape[0], batch_size):
        chunk = X[start : start + batch_size]
        if scipy.sparse.issparse(chunk):
            n_cells_total += np.array(chunk.getnnz(axis=0))
            gene_sum_total += np.array(chunk.sum(axis=0)).flatten()
        else:
            n_cells_total += np.count_nonzero(chunk, axis=0)
            gene_sum_total += chunk.sum(axis=0)
    return n_cells_total, gene_sum_total


def filter_genes(adata, cell_count_cutoff=15, cell_percentage_cutoff2=0.05, nonz_mean_cutoff=1.12):
    """Plot gene filter and return selected gene names.

    Copied from cell2location / regularizedvi to avoid a package dependency.
    """
    n_cells, gene_sum = _compute_gene_stats(adata.X)
    adata.var["n_cells"] = n_cells
    adata.var["nonz_mean"] = gene_sum / np.where(n_cells > 0, n_cells, 1)

    cell_count_cutoff = np.log10(cell_count_cutoff)
    cell_count_cutoff2 = np.log10(adata.shape[0] * cell_percentage_cutoff2)
    nonz_mean_cutoff = np.log10(nonz_mean_cutoff)

    gene_selection = (np.array(np.log10(adata.var["n_cells"]) > cell_count_cutoff2)) | (
        np.array(np.log10(adata.var["n_cells"]) > cell_count_cutoff)
        & np.array(np.log10(adata.var["nonz_mean"]) > nonz_mean_cutoff)
    )
    gene_selection = adata.var_names[gene_selection]
    adata_shape = adata[:, gene_selection].shape

    fig, ax = plt.subplots()
    ax.hist2d(
        np.log10(adata.var["nonz_mean"]),
        np.log10(adata.var["n_cells"]),
        bins=100,
        norm=matplotlib.colors.LogNorm(),
        range=[[0, 0.5], [1, 4.5]],
    )
    ax.axvspan(0, nonz_mean_cutoff, ymin=0.0, ymax=(cell_count_cutoff2 - 1) / 3.5, color="darkorange", alpha=0.3)
    ax.axvspan(
        nonz_mean_cutoff,
        np.max(np.log10(adata.var["nonz_mean"])),
        ymin=0.0,
        ymax=(cell_count_cutoff - 1) / 3.5,
        color="darkorange",
        alpha=0.3,
    )
    plt.vlines(nonz_mean_cutoff, cell_count_cutoff, cell_count_cutoff2, color="darkorange")
    plt.hlines(cell_count_cutoff, nonz_mean_cutoff, 1, color="darkorange")
    plt.hlines(cell_count_cutoff2, 0, nonz_mean_cutoff, color="darkorange")
    plt.xlabel("Mean non-zero expression level of gene (log)")
    plt.ylabel("Number of cells expressing gene (log)")
    plt.title(f"Gene filter: {adata_shape[0]} cells x {adata_shape[1]} genes")
    plt.show()

    return gene_selection


# ---------------------------------------------------------------------------
# Dataset loaders — each returns standardized AnnData
# ---------------------------------------------------------------------------


def load_bone_marrow(data_folder: str = "data/") -> sc.AnnData:
    """Load the NeurIPS 2021 bone marrow multiome dataset (GEX only).

    Downloads the h5ad and fragment files via ``download_bone_marrow_dataset``,
    filters to GEX features, sets ENSEMBL IDs as var_names, and standardizes obs.

    Parameters
    ----------
    data_folder
        Local directory for storing the downloaded files.

    Returns
    -------
    Standardized AnnData with raw GEX counts.
    """
    h5ad_path = download_bone_marrow_dataset(data_folder=data_folder)
    data_folder_abs = str(Path(data_folder).resolve())
    adata = sc.read_h5ad(h5ad_path)

    # Keep only Gene Expression features
    adata = adata[:, adata.var["feature_types"] == "GEX"].copy()

    # ENSEMBL IDs as var_names, gene symbols in var["SYMBOL"]
    adata = _set_ensembl_var_names(adata)

    # Set obs columns
    adata.obs["batch"] = adata.obs["batch"].values
    adata.obs["site"] = adata.obs["Site"].values
    adata.obs["donor"] = adata.obs["DonorNumber"].values
    adata.obs["dataset"] = "bone_marrow"
    adata.obs["tissue"] = "bone_marrow"
    adata.obs["condition"] = "healthy"
    adata.obs["age_group"] = "adult"
    adata.obs["sex"] = "unknown"
    adata.obs["original_annotation"] = adata.obs["l2_cell_type"].values
    adata.obs["harmonized_annotation"] = adata.obs["l2_cell_type"].map(BONE_MARROW_MAP).values
    adata.obs["fragment_file_path"] = [
        os.path.join(data_folder_abs, b, "atac_fragments.tsv.gz") for b in adata.obs["batch"]
    ]

    adata = _finalize(adata)

    # Print dataset summary
    print(f"Bone marrow dataset: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"  Batches:  {adata.obs['batch'].nunique()} ({adata.obs['batch'].unique().tolist()[:5]}...)")
    print(f"  Sites:    {adata.obs['site'].nunique()}")
    print(f"  Donors:   {adata.obs['donor'].nunique()}")
    print(f"  Cell types (harmonized): {adata.obs['harmonized_annotation'].nunique()}")

    return adata


def load_neat_seq_cd4t() -> sc.AnnData:
    """Load NEAT-seq CD4+ memory T cell dataset (8,457 cells x 36,717 genes).

    Source: Chen et al. NEAT-seq PBMC, CD4+ T cells subset.
    Two lanes (lane1, lane2), 7 clusters (C1-C7).
    """
    h5ad_path = "/nfs/team283/vk7/sanger_projects/large_data/neat_seq_pbmc/neat_seq_cd4_tcells.h5ad"
    frag_lane1 = "/nfs/team283/vk7/sanger_projects/large_data/neat_seq_pbmc/cd4_tcells/lane1/GSM5396332_lane1_atac_fragments.tsv.gz"
    frag_lane2 = "/nfs/team283/vk7/sanger_projects/large_data/neat_seq_pbmc/cd4_tcells/lane2/GSM5396336_lane2_atac_fragments.tsv.gz"

    adata = sc.read_h5ad(h5ad_path)

    # Swap var_names from gene symbols to ENSEMBL IDs
    adata.var["SYMBOL"] = adata.var_names.values.copy()
    adata.var_names = adata.var["gene_ids"].values.copy()
    drop_cols = [c for c in ["gene_ids", "gene_names", "feature_type"] if c in adata.var.columns]
    adata.var.drop(columns=drop_cols, inplace=True)
    adata.var_names_make_unique()

    # Standardized obs columns
    adata.obs["batch"] = ("neat_seq_" + adata.obs["lane"].astype(str)).values
    adata.obs["site"] = "stanford"
    adata.obs["donor"] = "neat_seq_donor1"
    adata.obs["dataset"] = "neat_seq_cd4t"
    adata.obs["tissue"] = "sorted_cd4t"
    adata.obs["condition"] = "healthy"
    adata.obs["age_group"] = "adult"
    adata.obs["sex"] = "unknown"
    adata.obs["original_annotation"] = adata.obs["Clusters"].values
    adata.obs["harmonized_annotation"] = adata.obs["Clusters"].map(NEAT_SEQ_MAP).values
    adata.obs["fragment_file_path"] = adata.obs["lane"].map({"lane1": frag_lane1, "lane2": frag_lane2}).values

    adata = _finalize(adata)
    return adata


def load_tea_seq_pbmc() -> sc.AnnData:
    """Load TEA-seq PBMC data from 7 samples (5 TEA-seq + 2 multiome).

    Reads sample_mapping.csv to locate per-sample H5 files and fragment files.
    Annotations (predicted.celltype.l2) are available only for GSM4949911 via
    the Figure4 supplementary CSV; other samples get NaN annotations.

    Returns
    -------
    Standardized AnnData with raw GEX counts.
    """
    sample_mapping_path = "/nfs/team283/vk7/sanger_projects/large_data/tea_seq_pbmc/sample_mapping.csv"
    annotation_path = (
        "/nfs/team283/vk7/sanger_projects/large_data/tea_seq_pbmc/"
        "supplementary_data/Figure4_SourceData2_TypeLabelsUMAP.csv"
    )

    # Read sample mapping
    sample_df = pd.read_csv(sample_mapping_path)

    # Load and process each sample
    adatas = []
    for _, row in sample_df.iterrows():
        sample_id = row["sample_id"]
        rna_h5_path = row["rna_h5_path"]
        fragment_file = row["fragment_file_path"]

        # Read 10x H5 and filter to Gene Expression
        adata_i = sc.read_10x_h5(rna_h5_path)
        adata_i = adata_i[:, adata_i.var["feature_types"] == "Gene Expression"].copy()

        # Set ENSEMBL as var_names
        adata_i = _set_ensembl_var_names(adata_i)

        # Prefix obs_names with sample_id
        adata_i.obs_names = [f"{sample_id}#{bc}" for bc in adata_i.obs_names]

        # Store fragment file path and sample_id per cell
        adata_i.obs["fragment_file_path"] = fragment_file
        adata_i.obs["sample_id"] = sample_id

        adatas.append(adata_i)

    # Concatenate all samples (inner join on genes)
    adata = anndata.concat(adatas, join="inner")
    del adatas
    gc.collect()

    # Load annotations for GSM4949911
    annot_df = pd.read_csv(annotation_path)
    annot_df = annot_df[["barcode", "predicted.celltype.l2"]].copy()
    annot_df = annot_df.set_index("barcode")

    # Map annotations: extract raw barcode from obs_names for GSM4949911 cells
    gsm4949911_mask = adata.obs["sample_id"] == "GSM4949911_tea_seq"
    raw_barcodes = pd.Series(
        [name.split("#", 1)[1] for name in adata.obs_names[gsm4949911_mask]],
        index=adata.obs_names[gsm4949911_mask],
    )
    # Look up annotations by raw barcode
    annotations = raw_barcodes.map(annot_df["predicted.celltype.l2"])

    # Set obs columns
    adata.obs["batch"] = adata.obs["sample_id"].values
    adata.obs["site"] = "allen_institute"
    adata.obs["donor"] = "pbmc_donor1"
    adata.obs["dataset"] = "pbmc_tea_seq"
    adata.obs["tissue"] = "pbmc"
    adata.obs["condition"] = "healthy"
    adata.obs["age_group"] = "adult"
    adata.obs["sex"] = "unknown"

    # original_annotation: only available for GSM4949911
    adata.obs["original_annotation"] = np.nan
    adata.obs.loc[gsm4949911_mask, "original_annotation"] = annotations.values

    # harmonized_annotation via TEA_SEQ_MAP
    adata.obs["harmonized_annotation"] = adata.obs["original_annotation"].map(TEA_SEQ_MAP)

    # Clean up temporary column
    adata.obs.drop(columns=["sample_id"], inplace=True)

    return _finalize(adata)


def load_covid_pbmc_gse239799() -> sc.AnnData:
    """Load COVID infant PBMC multiome dataset GSE239799 (GEX only).

    43 samples from 18 subjects (longitudinal). No cell annotations available.

    Returns
    -------
    Standardized AnnData with raw GEX counts.
    """
    csv_path = "/nfs/team283/vk7/sanger_projects/large_data/multiome_pbmc/GSE239799/sample_mapping.csv"
    mapping = pd.read_csv(csv_path)

    adatas = []
    for _, row in mapping.iterrows():
        ad = _read_10x_mtx(
            barcodes_path=row["rna_barcodes_path"],
            features_path=row["rna_features_path"],
            matrix_path=row["rna_mtx_path"],
            sample_id=row["sample_id"],
        )
        adatas.append(ad)
        del ad
        gc.collect()

    adata = anndata.concat(adatas, join="inner")
    del adatas
    gc.collect()

    # Build sample-level lookup from mapping
    sample_to_subject = dict(zip(mapping["sample_id"], mapping["subject_id"], strict=True))
    sample_to_frag = dict(zip(mapping["sample_id"], mapping["fragment_file_path"], strict=True))

    # Set obs columns
    adata.obs["batch"] = adata.obs["sample"].values
    adata.obs["site"] = "stanford_wimmers"
    adata.obs["donor"] = adata.obs["sample"].map(sample_to_subject).values
    adata.obs["dataset"] = "covid_pbmc"
    adata.obs["tissue"] = "pbmc"
    adata.obs["condition"] = "covid"
    adata.obs["age_group"] = "child"
    adata.obs["sex"] = "unknown"
    adata.obs["original_annotation"] = np.nan
    adata.obs["harmonized_annotation"] = np.nan
    adata.obs["fragment_file_path"] = adata.obs["sample"].map(sample_to_frag).values

    adata = _finalize(adata)

    # Print dataset summary
    print(f"COVID PBMC GSE239799 dataset: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"  Samples: {adata.obs['batch'].nunique()}")
    print(f"  Donors:  {adata.obs['donor'].nunique()}")

    return adata


def load_infant_adult_spleen_gse311423() -> sc.AnnData:
    """Load infant/adult spleen multiome data from GSE311423 (GEX only).

    Loads 5 samples (3 infant, 2 adult) from the sample mapping CSV,
    filters to Gene Expression features, sets ENSEMBL IDs as var_names,
    and standardizes obs.

    There are no cell annotations for this dataset.

    Returns
    -------
    Standardized AnnData with raw GEX counts.
    """
    sample_csv = "/nfs/team283/vk7/sanger_projects/large_data/multiome_spleen_lung/GSE311423/sample_mapping.csv"
    sample_df = pd.read_csv(sample_csv)

    adatas = []
    for _, row in sample_df.iterrows():
        adata_i = sc.read_10x_h5(row["gex_h5_path"])

        # Keep only Gene Expression features
        adata_i = adata_i[:, adata_i.var["feature_types"] == "Gene Expression"].copy()

        # ENSEMBL IDs as var_names, gene symbols in var["SYMBOL"]
        adata_i = _set_ensembl_var_names(adata_i)

        # Prefix obs_names with library_id
        library_id = row["library_id"]
        adata_i.obs_names = [f"{library_id}#{bc}" for bc in adata_i.obs_names]

        # Store per-sample metadata for obs columns after concat
        adata_i.obs["batch"] = library_id
        adata_i.obs["donor"] = row["donor_id"]
        adata_i.obs["age_group"] = row["age_group"]
        adata_i.obs["fragment_file_path"] = row["fragment_file_path"]

        adatas.append(adata_i)

    adata = anndata.concat(adatas, join="inner")

    # Set obs columns
    adata.obs["site"] = "columbia"
    adata.obs["dataset"] = "infant_adult_spleen"
    adata.obs["tissue"] = "spleen"
    adata.obs["condition"] = "healthy"
    adata.obs["sex"] = "unknown"
    adata.obs["original_annotation"] = np.nan
    adata.obs["harmonized_annotation"] = np.nan

    adata = _finalize(adata)

    # Print dataset summary
    print(f"Infant/adult spleen (GSE311423): {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"  Batches:    {adata.obs['batch'].nunique()} ({adata.obs['batch'].unique().tolist()})")
    print(f"  Donors:     {adata.obs['donor'].nunique()} ({adata.obs['donor'].unique().tolist()})")
    print(f"  Age groups: {adata.obs['age_group'].unique().tolist()}")

    return adata


def load_crohns_pbmc_gse244831() -> sc.AnnData:
    """Load Crohn's disease PBMC multiome dataset (GEX only) from GSE244831.

    Loads 13 samples from 10x MTX files, joins cell-level annotations, and
    standardizes obs columns.

    Returns
    -------
    Standardized AnnData with raw GEX counts.
    """
    base = "/nfs/team283/vk7/sanger_projects/large_data/multiome_pbmc/GSE244831"
    sample_mapping = pd.read_csv(os.path.join(base, "sample_mapping.csv"))

    # Load each sample using the shared MTX reader
    adatas = []
    for _, row in sample_mapping.iterrows():
        print(f"  Loading {row['sample_id']}...")
        ad = _read_10x_mtx(
            row["barcodes_path"],
            row["features_path"],
            row["matrix_path"],
            row["sample_id"],
        )
        adatas.append(ad)

    # Concatenate all samples (inner join keeps shared genes only)
    adata = anndata.concat(adatas, join="inner")
    del adatas
    gc.collect()
    print(f"  Concatenated: {adata.n_obs} cells x {adata.n_vars} genes")

    # Load cell annotations
    annot = pd.read_csv(
        os.path.join(base, "annotations", "GSE244831_cell_annotations.csv"),
    )
    annot.index = annot["barcode"].values
    annot = annot.loc[annot.index.isin(adata.obs_names)]

    # Join annotations to adata.obs
    adata.obs = adata.obs.join(annot, how="left")

    # Build fragment file path lookup (sample_id -> path)
    frag_map = dict(zip(sample_mapping["sample_id"], sample_mapping["fragment_file_path"], strict=True))

    # Set standardized obs columns
    adata.obs["batch"] = adata.obs["Sample"].values
    adata.obs["site"] = "emory"
    adata.obs["donor"] = adata.obs["Donors_IDs"].values
    adata.obs["dataset"] = "crohns_pbmc"
    adata.obs["tissue"] = "pbmc"
    adata.obs["condition"] = adata.obs["Status"].str.lower().values
    adata.obs["age_group"] = "adult"
    adata.obs["sex"] = adata.obs["Sex"].values
    adata.obs["original_annotation"] = adata.obs["Celltypes"].values
    adata.obs["harmonized_annotation"] = adata.obs["Celltypes"].map(CROHNS_MAP).values
    adata.obs["fragment_file_path"] = [frag_map.get(s, np.nan) for s in adata.obs["sample"].values]

    adata = _finalize(adata)

    # Print dataset summary
    print(f"Crohn's PBMC dataset: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"  Batches:  {adata.obs['batch'].nunique()} ({adata.obs['batch'].unique().tolist()[:5]}...)")
    print(f"  Sites:    {adata.obs['site'].nunique()}")
    print(f"  Donors:   {adata.obs['donor'].nunique()}")
    print(f"  Conditions: {adata.obs['condition'].unique().tolist()}")
    print(f"  Cell types (harmonized): {adata.obs['harmonized_annotation'].nunique()}")

    return adata


def load_lung_spleen_gse319044() -> sc.AnnData:
    """Load lung and spleen immune multiome data (GEX only) from GSE319044.

    Loads 16 samples (9 lung + 7 spleen) from 10x multiome experiments,
    reads GEX counts from MTX files, and joins with cell-type annotations.

    Returns
    -------
    Standardized AnnData with raw GEX counts.
    """
    # --- 1. Read sample mapping ---
    sample_mapping_path = (
        "/nfs/team283/vk7/sanger_projects/large_data/multiome_spleen_lung/GSE319044/sample_mapping.csv"
    )
    sample_df = pd.read_csv(sample_mapping_path)

    # Build lookup dicts from sample_id
    sample_to_donor = dict(zip(sample_df["sample_id"], sample_df["file_donor_id"], strict=True))
    sample_to_tissue = dict(zip(sample_df["sample_id"], sample_df["tissue"], strict=True))
    sample_to_fragment = dict(zip(sample_df["sample_id"], sample_df["fragment_file_path"], strict=True))

    # --- 2. Read 10x MTX for each sample ---
    adatas = []
    for _, row in sample_df.iterrows():
        ad = _read_10x_mtx(
            barcodes_path=row["barcodes_path"],
            features_path=row["features_path"],
            matrix_path=row["matrix_path"],
            sample_id=row["sample_id"],
        )
        adatas.append(ad)
        print(f"  Loaded {row['sample_id']}: {ad.n_obs} cells x {ad.n_vars} genes")

    # --- 3. Concatenate all samples ---
    adata = anndata.concat(adatas, join="inner")
    del adatas
    gc.collect()
    print(f"Concatenated: {adata.n_obs} cells x {adata.n_vars} genes")

    # --- 4. Load annotation CSV and match barcodes ---
    annotation_path = (
        "/nfs/team283/vk7/sanger_projects/large_data/multiome_spleen_lung"
        "/GSE319044/series_level/GSE319044_snRNA_cluster_labels.csv.gz"
    )
    ann = pd.read_csv(annotation_path, index_col=0)

    # Annotation barcodes have Seurat-style suffixes (e.g. AAACAGCCAACAACAA-2_1).
    # The 10x barcodes in adata are {sample_id}#{barcode} where barcode is the raw
    # 10x barcode (e.g. AAACAGCCAACAACAA-1). We match via:
    #   1. Extract the 16-mer nucleotide sequence from the annotation barcode
    #   2. Use annotation library_id + tissue.ident to find the sample_id
    #   3. Reconstruct as {sample_id}#{nucleotide_seq}-1

    # Reverse lookup: (library_id, tissue) -> sample_id
    # The annotation CSV library_id values (e.g. SMO-5, SMO-9) may differ from
    # sample_mapping file_donor_id (e.g. SMO-05, SMO-09) — strip leading zeros.
    def _normalize_donor(d):
        """SMO-05 -> SMO-5, COB-11 -> COB-11"""
        return re.sub(r"-0+(\d)", r"-\1", str(d))

    donor_tissue_to_sample = {}
    for _, row in sample_df.iterrows():
        norm = _normalize_donor(row["file_donor_id"])
        donor_tissue_to_sample[(norm, row["tissue"])] = row["sample_id"]
        # Also index with original in case they match exactly
        donor_tissue_to_sample[(row["file_donor_id"], row["tissue"])] = row["sample_id"]

    # Also build donor-only lookup for fallback
    donor_to_samples = {}
    for _, row in sample_df.iterrows():
        norm = _normalize_donor(row["file_donor_id"])
        donor_to_samples.setdefault(norm, []).append(row["sample_id"])
        donor_to_samples.setdefault(row["file_donor_id"], []).append(row["sample_id"])

    # Map annotation tissue.ident to sample_mapping tissue values
    tissue_norm = {
        "lungs": "lung",
        "Lung": "lung",
        "spleen": "spleen",
        "spleens": "spleen",
        "Spleen": "spleen",
    }

    # Extract nucleotide base from annotation barcodes
    barcode_base = ann.index.to_series().str.extract(r"^([ACGT]+)", expand=False)

    # Build new index for annotation rows
    ann_library = ann["library_id"].values
    ann_tissue_raw = ann["tissue.ident"].values
    new_index = []
    for i in range(len(ann)):
        lib_id = ann_library[i]
        tissue_clean = tissue_norm.get(str(ann_tissue_raw[i]), str(ann_tissue_raw[i]))
        key = (lib_id, tissue_clean)
        sample_id = donor_tissue_to_sample.get(key)
        if sample_id is None:
            # Fallback: match by donor only (if unique)
            candidates = donor_to_samples.get(lib_id, [])
            if len(candidates) == 1:
                sample_id = candidates[0]
        nuc = barcode_base.iloc[i]
        if sample_id is not None and pd.notna(nuc):
            new_index.append(f"{sample_id}#{nuc}-1")
        else:
            new_index.append(None)

    ann.index = new_index

    # Drop unmapped rows and duplicates
    ann = ann.loc[ann.index.dropna()]
    ann = ann[~ann.index.duplicated(keep="first")]

    # --- 5. Join annotation to adata.obs ---
    common_cells = adata.obs_names.intersection(ann.index)
    print(f"Annotation match: {len(common_cells)} / {adata.n_obs} cells ({len(common_cells) / adata.n_obs * 100:.1f}%)")

    ann_matched = ann.loc[common_cells]

    # Set obs columns
    adata.obs["batch"] = adata.obs["sample"].values
    adata.obs["site"] = "uchicago"
    adata.obs["donor"] = adata.obs["sample"].map(sample_to_donor).values
    adata.obs["dataset"] = "lung_spleen_gse319044"
    adata.obs["tissue"] = adata.obs["sample"].map(sample_to_tissue).values
    adata.obs["fragment_file_path"] = adata.obs["sample"].map(sample_to_fragment).values

    # Annotation-dependent columns (NaN for unannotated cells)
    adata.obs["original_annotation"] = np.nan
    adata.obs.loc[common_cells, "original_annotation"] = ann_matched["CellType"].values

    adata.obs["harmonized_annotation"] = np.nan
    adata.obs.loc[common_cells, "harmonized_annotation"] = ann_matched["CellType"].map(LUNG_SPLEEN_MAP).values

    # Condition from Asthmatic.status
    asthmatic_map = {
        "non": "non_asthmatic",
        "Non": "non_asthmatic",
        "non-asthmatic": "non_asthmatic",
        "Non-Asthmatic": "non_asthmatic",
        "Non-asthmatic": "non_asthmatic",
        "asthmatic": "asthmatic",
        "Asthmatic": "asthmatic",
        "yes": "asthmatic",
        "Yes": "asthmatic",
    }
    adata.obs["condition"] = "unknown"
    if "Asthmatic.status" in ann_matched.columns:
        mapped_condition = ann_matched["Asthmatic.status"].map(asthmatic_map)
        mapped_condition = mapped_condition.fillna("unknown")
        adata.obs.loc[common_cells, "condition"] = mapped_condition.values

    adata.obs["age_group"] = "adult"

    # Sex from annotation
    adata.obs["sex"] = "unknown"
    if "Sex" in ann_matched.columns:
        sex_values = ann_matched["Sex"].str.lower().fillna("unknown")
        adata.obs.loc[common_cells, "sex"] = sex_values.values

    # --- 6. Finalize ---
    adata = _finalize(adata)

    # Print dataset summary
    print(f"Lung/spleen GSE319044 dataset: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"  Batches:  {adata.obs['batch'].nunique()} ({adata.obs['batch'].unique().tolist()[:5]}...)")
    print(f"  Sites:    {adata.obs['site'].nunique()}")
    print(f"  Donors:   {adata.obs['donor'].nunique()}")
    print(f"  Tissues:  {adata.obs['tissue'].value_counts().to_dict()}")
    print(f"  Cell types (harmonized): {adata.obs['harmonized_annotation'].nunique()}")
    print(f"  Annotated cells: {adata.obs['original_annotation'].notna().sum()} / {adata.n_obs}")

    return adata
