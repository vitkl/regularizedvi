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
import warnings
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
    "level_1",
    "level_2",
    "level_3",
    "level_4",
    "fragment_file_path",
]

_LEVEL_PREFIX = "level_1:"

# Harmonization mappings are loaded from annotation_harmonization.md (single
# source of truth).  Edit the markdown file to change mappings — the Python
# code reads it at runtime via _get_harmonization_maps().

_HARMONIZATION_MAPS = None
_HIERARCHY_DF = None
_VALIDATED = False


def _parse_harmonization_md(md_path):
    """Parse annotation_harmonization.md into per-dataset mapping dicts.

    Parameters
    ----------
    md_path : str or Path
        Path to annotation_harmonization.md.

    Returns
    -------
    dict[str, dict[str, str]]
        ``{source_dataset: {original_label: harmonized_name}}``.
        Skips section headers (bold text) and entries with empty
        harmonized_name.
    """
    maps = {}
    with open(md_path) as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.split("|")]
            # split on "|" gives empty strings at start/end
            cells = [c for c in cells if c != ""]
            if len(cells) < 4:
                continue
            original, harmonized, dataset, _col = cells[0], cells[1], cells[2], cells[3]
            # Skip header row, separator row, section headers, and notes
            if original in ("original_label", "---") or original.startswith("**") or original.startswith("("):
                continue
            if dataset.startswith("---"):
                continue
            if not harmonized or not dataset or dataset == "—":
                continue
            # Convert literal "NaN" to np.nan so .map() produces NaN
            if harmonized == "NaN":
                maps.setdefault(dataset, {})[original] = np.nan
            else:
                maps.setdefault(dataset, {})[original] = harmonized
    return maps


def _parse_hierarchy_md(md_path):
    """Parse annotation_hierarchy.md into a DataFrame.

    Parameters
    ----------
    md_path : str or Path
        Path to annotation_hierarchy.md.

    Returns
    -------
    pd.DataFrame
        Columns: harmonized_name, level_1, level_2, level_3, level_4.
        Skips section headers (bold rows with empty level columns).
    """
    rows = []
    with open(md_path) as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.split("|")]
            cells = [c for c in cells if c != ""]
            if len(cells) < 5:
                continue
            name, l1, l2, l3, l4 = cells[0], cells[1], cells[2], cells[3], cells[4]
            # Skip header, separator, and section headers (bold with empty levels)
            if name in ("harmonized_name", "---") or name.startswith("**"):
                continue
            if l1.startswith("---"):
                continue
            rows.append({"harmonized_name": name, "level_1": l1, "level_2": l2, "level_3": l3, "level_4": l4})
    return pd.DataFrame(rows)


def _validate_annotations(harm_maps, hier_df):
    """Cross-validate harmonization maps against hierarchy DataFrame.

    Raises ValueError on fatal issues, emits warnings for non-fatal ones.
    """
    errors = []

    # Collect all harmonized names from harmonization (across all datasets)
    all_harm_names = set()
    level1_prefix_values = set()
    for dataset_map in harm_maps.values():
        for harmonized in dataset_map.values():
            if isinstance(harmonized, float) and np.isnan(harmonized):
                continue
            if isinstance(harmonized, str) and harmonized.startswith(_LEVEL_PREFIX):
                level1_prefix_values.add(harmonized[len(_LEVEL_PREFIX) :])
            else:
                all_harm_names.add(harmonized)

    hier_names = set(hier_df["harmonized_name"])
    hier_level1_values = set(hier_df["level_1"])

    # (e) No duplicate harmonized_name in hierarchy
    dupes = hier_df["harmonized_name"][hier_df["harmonized_name"].duplicated()].unique()
    if len(dupes) > 0:
        errors.append(f"Duplicate harmonized_name in hierarchy: {sorted(dupes)}")

    # (a) All harmonized names from harmonization exist in hierarchy
    missing_from_hier = all_harm_names - hier_names
    if missing_from_hier:
        errors.append(f"Harmonized names not found in hierarchy: {sorted(missing_from_hier)}")

    # (c) All level_1: prefix values exist as level_1 in hierarchy
    missing_l1 = level1_prefix_values - hier_level1_values
    if missing_l1:
        errors.append(f"level_1: prefix values not found in hierarchy level_1: {sorted(missing_l1)}")

    # (f) Consistent parents for shared level_1 values
    for l1_val, group in hier_df.groupby("level_1"):
        for col in ["level_2", "level_3", "level_4"]:
            unique_vals = group[col].unique()
            if len(unique_vals) > 1:
                errors.append(f"Inconsistent {col} for level_1='{l1_val}': {sorted(unique_vals)}")

    if errors:
        raise ValueError("Annotation validation failed:\n  - " + "\n  - ".join(errors))

    # (b) Unreferenced hierarchy entries (WARNING)
    unreferenced = hier_names - all_harm_names
    if unreferenced:
        warnings.warn(
            f"Hierarchy entries not referenced by any harmonization mapping: "
            f"{sorted(unreferenced)}. These may be kept for future datasets.",
            UserWarning,
            stacklevel=3,
        )

    # (d) Higher-level names not at lower levels (WARNING)
    level_cols = ["level_1", "level_2", "level_3", "level_4"]
    cross_level_warnings = []
    for coarser_idx in range(3, 0, -1):
        coarser_col = level_cols[coarser_idx]
        coarser_values = set(hier_df[coarser_col])
        for finer_idx in range(coarser_idx):
            finer_col = level_cols[finer_idx]
            finer_values = set(hier_df[finer_col])
            overlap = coarser_values & finer_values
            if overlap:
                cross_level_warnings.append(f"{coarser_col} values also in {finer_col}: {sorted(overlap)}")
    if cross_level_warnings:
        warnings.warn(
            "Cross-level name overlap (structural collapses, not necessarily errors):\n  - "
            + "\n  - ".join(cross_level_warnings),
            UserWarning,
            stacklevel=3,
        )


def _get_validated_data():
    """Load, validate, and cache both harmonization maps and hierarchy."""
    global _HARMONIZATION_MAPS, _HIERARCHY_DF, _VALIDATED
    if not _VALIDATED:
        md_dir = Path(__file__).parent
        _HARMONIZATION_MAPS = _parse_harmonization_md(md_dir / "annotation_harmonization.md")
        _HIERARCHY_DF = _parse_hierarchy_md(md_dir / "annotation_hierarchy.md")
        _validate_annotations(_HARMONIZATION_MAPS, _HIERARCHY_DF)
        _VALIDATED = True
    return _HARMONIZATION_MAPS, _HIERARCHY_DF


def _get_harmonization_maps():
    """Load and cache harmonization maps from annotation_harmonization.md."""
    return _get_validated_data()[0]


def _get_hierarchy_df():
    """Load and cache hierarchy from annotation_hierarchy.md."""
    return _get_validated_data()[1]


def _apply_hierarchy(adata, hier_df=None):
    """Populate level_1..level_4 from harmonized_annotation.

    Handles three cases:
    1. Normal harmonized names -> look up in hierarchy DF
    2. ``level_1:`` prefixed strings -> set harmonized_annotation=NaN,
       populate levels by looking up the level_1 value in hierarchy
    3. NaN -> all levels stay NaN
    """
    if hier_df is None:
        hier = _get_hierarchy_df()
    else:
        hier = hier_df
    hier_indexed = hier.set_index("harmonized_name")
    level_cols = ["level_1", "level_2", "level_3", "level_4"]

    # Build level_1 -> {level_1, level_2, level_3, level_4} lookup
    # (validation check f guarantees consistent parents per level_1)
    l1_to_levels = {}
    for l1_val, group in hier.groupby("level_1"):
        row = group.iloc[0]
        l1_to_levels[l1_val] = {
            "level_1": l1_val,
            "level_2": row["level_2"],
            "level_3": row["level_3"],
            "level_4": row["level_4"],
        }

    ha = adata.obs["harmonized_annotation"].astype(object)

    # Detect level_1: prefixed entries
    is_prefix = ha.str.startswith(_LEVEL_PREFIX, na=False)

    # Initialize level columns as object dtype (avoids FutureWarning on mixed str/NaN)
    for col in level_cols:
        adata.obs[col] = pd.array([np.nan] * adata.n_obs, dtype=object)

    # Case 1: Normal harmonized names (not NaN, not prefixed)
    normal_mask = ha.notna() & ~is_prefix
    if normal_mask.any():
        normal_names = ha[normal_mask]
        for col in level_cols:
            col_map = hier_indexed[col].to_dict()
            adata.obs.loc[normal_mask, col] = normal_names.map(col_map).values

    # Case 2: level_1: prefixed entries
    if is_prefix.any():
        prefix_vals = ha[is_prefix].str[len(_LEVEL_PREFIX) :]
        for col in level_cols:
            col_map = {k: v[col] for k, v in l1_to_levels.items()}
            adata.obs.loc[is_prefix, col] = prefix_vals.map(col_map).values
        # Set harmonized_annotation to NaN for prefixed entries
        adata.obs.loc[is_prefix, "harmonized_annotation"] = np.nan

    # Case 3: NaN stays NaN (already initialized above)

    return adata


def update_annotations(adata, rename_map=None, add_hierarchy=False, hierarchy_md_path=None):
    """Update harmonized annotations and optionally add hierarchy levels.

    Parameters
    ----------
    adata : AnnData
        Must have ``obs["harmonized_annotation"]``.
    rename_map : dict, optional
        ``{old_name: new_name}`` for renaming cell types.
    add_hierarchy : bool
        If True, join hierarchy levels from annotation_hierarchy.md.
    hierarchy_md_path : str or Path, optional
        Override path to hierarchy markdown.  Defaults to
        ``annotation_hierarchy.md`` next to this file.

    Returns
    -------
    AnnData
        adata with updated obs columns (modified in-place and returned).
    """
    if rename_map is not None:
        adata.obs["harmonized_annotation"] = adata.obs["harmonized_annotation"].map(lambda x: rename_map.get(x, x))
    if add_hierarchy:
        if hierarchy_md_path is not None:
            hier = _parse_hierarchy_md(hierarchy_md_path)
            _apply_hierarchy(adata, hier_df=hier)
        else:
            _apply_hierarchy(adata)
    return adata


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

    # Prefix barcodes with sample_id: {sample_id}-{barcode} (cell2state format)
    adata.obs_names = [f"{sample_id}-{bc}" for bc in adata.obs_names]
    adata.obs["sample"] = sample_id

    return adata


def _finalize(adata: sc.AnnData) -> sc.AnnData:
    """Final steps: validate integer counts, ensure sparse X, apply hierarchy, standardize obs.

    Keeps source dtypes (no uint16 cast) — downstream notebooks cast before saving.
    """
    # Validate that X contains integer counts (not normalized data)
    if scipy.sparse.issparse(adata.X):
        sample_vals = adata.X.data[: min(10000, len(adata.X.data))]
    else:
        sample_vals = np.asarray(adata.X).ravel()[:10000]
    if len(sample_vals) > 0 and not np.allclose(sample_vals, np.round(sample_vals), atol=1e-3):
        raise ValueError(
            "adata.X contains non-integer values — likely normalized data. "
            "Set adata.X = adata.layers['counts'] before calling _finalize()."
        )
    # Ensure sparse CSR format but keep source dtype
    if scipy.sparse.issparse(adata.X):
        adata.X = adata.X.tocsr()
    else:
        adata.X = scipy.sparse.csr_matrix(adata.X)
    adata.layers["counts"] = adata.X.copy()
    adata = _apply_hierarchy(adata)
    adata = _standardize_obs(adata)
    return adata


# ---------------------------------------------------------------------------
# CSV loading utilities (safe reindex, never .obs.join)
# ---------------------------------------------------------------------------


def load_scrublet_scores(adata: sc.AnnData, csv_path: str) -> sc.AnnData:
    """Load scrublet CSV, reindex to adata.obs_names, fill NaN with 0.0."""
    df = pd.read_csv(csv_path, index_col=0)
    df["predicted_doublet"] = df["predicted_doublet"].map({"True": True, "False": False, True: True, False: False})
    df["doublet_score"] = df["doublet_score"].astype("float32")
    reindexed = df.reindex(index=adata.obs_names)
    n_matched = reindexed["doublet_score"].notna().sum()
    print(f"Scrublet: {n_matched}/{adata.n_obs} matched ({100 * n_matched / adata.n_obs:.1f}%)")
    if n_matched == 0:
        raise ValueError(
            f"No scrublet match. CSV index sample: {list(df.index[:3])}, obs_names sample: {list(adata.obs_names[:3])}"
        )
    # NaN → 0.0 (keeps cell — scrublet couldn't run for that batch)
    reindexed["doublet_score"] = reindexed["doublet_score"].fillna(0.0)
    reindexed["predicted_doublet"] = reindexed["predicted_doublet"].fillna(False)
    for col in df.columns:
        adata.obs[col] = reindexed[col].values
    return adata


def load_atac_qc_metrics(adata: sc.AnnData, csv_path: str) -> sc.AnnData:
    """Load ATAC QC CSV, reindex to adata.obs_names."""
    df = pd.read_csv(csv_path, index_col=0)
    reindexed = df.reindex(index=adata.obs_names)
    n_matched = reindexed.iloc[:, 0].notna().sum()
    print(f"ATAC QC: {n_matched}/{adata.n_obs} matched ({100 * n_matched / adata.n_obs:.1f}%)")
    if n_matched == 0:
        raise ValueError(
            f"No ATAC QC match. CSV index sample: {list(df.index[:3])}, obs_names sample: {list(adata.obs_names[:3])}"
        )
    for col in df.columns:
        adata.obs[col] = reindexed[col].values
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


def download_bone_marrow_dataset(data_folder: str = "/nfs/team283/vk7/sanger_projects/large_data/bone_marrow/") -> str:
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


def load_bone_marrow(data_folder: str = "/nfs/team283/vk7/sanger_projects/large_data/bone_marrow/") -> sc.AnnData:
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
    adata.obs["site"] = adata.obs["site"].values
    adata.obs["donor"] = adata.obs["donor"].values
    adata.obs["dataset"] = "bone_marrow"
    adata.obs["tissue"] = "bone_marrow"
    adata.obs["condition"] = "healthy"
    adata.obs["age_group"] = "adult"
    adata.obs["sex"] = "unknown"
    adata.obs["original_annotation"] = adata.obs["l2_cell_type"].values
    adata.obs["harmonized_annotation"] = adata.obs["l2_cell_type"].map(_get_harmonization_maps()["bone_marrow"]).values
    adata.obs["fragment_file_path"] = [
        os.path.join(data_folder_abs, b, "atac_fragments.tsv.gz") for b in adata.obs["batch"]
    ]

    # Raw counts are in layers["counts"]; X may contain normalized data
    adata.X = adata.layers["counts"]

    # Prefix obs_names with batch for {batch}-{barcode} format
    adata.obs_names = [
        f"{batch}-{bc}" if bc.endswith("-1") else f"{batch}-{bc}-1"
        for bc, batch in zip(adata.obs_names, adata.obs["batch"], strict=True)
    ]

    adata = _finalize(adata)

    # Print dataset summary
    print(f"Bone marrow dataset: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"  Batches:  {adata.obs['batch'].nunique()} ({adata.obs['batch'].unique().tolist()[:5]}...)")
    print(f"  Sites:    {adata.obs['site'].nunique()}")
    print(f"  Donors:   {adata.obs['donor'].nunique()}")
    print(f"  Cell types (harmonized): {adata.obs['harmonized_annotation'].nunique()}")

    return adata


def load_neat_seq_cd4t(
    data_folder: str = "/nfs/team283/vk7/sanger_projects/large_data/neat_seq_pbmc/",
) -> sc.AnnData:
    """Load NEAT-seq CD4+ memory T cell dataset (8,457 cells x 36,717 genes).

    Source: Chen et al. NEAT-seq PBMC, CD4+ T cells subset.
    Two lanes (lane1, lane2), 7 clusters (C1-C7).
    """
    h5ad_path = os.path.join(data_folder, "neat_seq_cd4_tcells.h5ad")
    frag_lane1 = os.path.join(data_folder, "cd4_tcells/lane1/GSM5396332_lane1_atac_fragments.tsv.gz")
    frag_lane2 = os.path.join(data_folder, "cd4_tcells/lane2/GSM5396336_lane2_atac_fragments.tsv.gz")

    adata = sc.read_h5ad(h5ad_path)

    # Swap var_names from gene symbols to ENSEMBL IDs
    adata.var["SYMBOL"] = adata.var_names.values.copy()
    adata.var_names = adata.var["gene_ids"].values.copy()
    drop_cols = [c for c in ["gene_ids", "gene_names", "feature_type"] if c in adata.var.columns]
    adata.var.drop(columns=drop_cols, inplace=True)
    adata.var_names_make_unique()

    # Standardized obs columns
    adata.obs["batch"] = ("neat_seq_" + adata.obs["lane"].astype(str)).values

    # Prefix obs_names: {batch}-{barcode} (cell2state format)
    adata.obs_names = [
        f"{batch}-{bc}" if bc.endswith("-1") else f"{batch}-{bc}-1"
        for bc, batch in zip(adata.obs_names, adata.obs["batch"], strict=True)
    ]

    adata.obs["site"] = "stanford"
    adata.obs["donor"] = "neat_seq_donor1"
    adata.obs["dataset"] = "neat_seq_cd4t"
    adata.obs["tissue"] = "sorted_cd4t"
    adata.obs["condition"] = "healthy"
    adata.obs["age_group"] = "adult"
    adata.obs["sex"] = "unknown"
    adata.obs["original_annotation"] = adata.obs["Clusters"].values
    adata.obs["harmonized_annotation"] = adata.obs["Clusters"].map(_get_harmonization_maps()["neat_seq_cd4t"]).values
    adata.obs["fragment_file_path"] = adata.obs["lane"].map({"lane1": frag_lane1, "lane2": frag_lane2}).values

    adata = _finalize(adata)
    return adata


def load_tea_seq_pbmc(
    data_folder: str = "/nfs/team283/vk7/sanger_projects/large_data/tea_seq_pbmc/",
) -> sc.AnnData:
    """Load TEA-seq PBMC data from 7 samples (5 TEA-seq + 2 multiome).

    Reads sample_mapping.csv to locate per-sample H5 files and fragment files.
    Annotations (predicted.celltype.l2) are available for 4 of 5 TEA-seq wells
    (GSM5123951-54) via the Figure4 supplementary CSV. GSM4949911 and multiome
    wells have no annotations.

    Returns
    -------
    Standardized AnnData with raw GEX counts.
    """
    sample_mapping_path = os.path.join(data_folder, "sample_mapping.csv")
    annotation_path = os.path.join(data_folder, "supplementary_data/Figure4_SourceData2_TypeLabelsUMAP.csv")

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

        # Prefix obs_names with sample_id: {sample_id}-{barcode} (cell2state format)
        adata_i.obs_names = [f"{sample_id}-{bc}" for bc in adata_i.obs_names]

        # Store fragment file path and sample_id per cell
        adata_i.obs["fragment_file_path"] = fragment_file
        adata_i.obs["sample_id"] = sample_id

        adatas.append(adata_i)

    # Concatenate all samples (inner join on genes)
    adata = anndata.concat(adatas, join="inner")
    del adatas
    gc.collect()

    # --- Annotation matching (BEFORE UUID mapping, since annotations use cellranger barcodes) ---
    # Load annotations from Figure4 CSV (covers GSM5123951-54 as suffixes 3-6)
    # GSM4949911 is NOT in this CSV. Barcodes have Seurat merge suffixes (-3 to -6)
    # while per-sample H5 files use -1. Match each well to its known suffix.
    annot_df = pd.read_csv(annotation_path)
    annot_df = annot_df[["barcode", "predicted.celltype.l2"]].copy()
    annot_df["bc_seq"] = annot_df["barcode"].str.rsplit("-", n=1).str[0]
    annot_df["suffix"] = annot_df["barcode"].str.rsplit("-", n=1).str[1]

    # Mapping: sample_id -> Seurat suffix in the Figure4 CSV
    well_to_suffix = {
        "GSM5123951_tea_seq": "3",
        "GSM5123952_tea_seq": "4",
        "GSM5123953_tea_seq": "5",
        "GSM5123954_tea_seq": "6",
    }

    # Build per-well annotation lookup (barcode sequence -> cell type)
    # obs_names are still {sample_id}-{cellranger_bc} at this point
    annotations = pd.Series(np.nan, index=adata.obs_names)
    for sample_id, suffix in well_to_suffix.items():
        well_mask = adata.obs["sample_id"] == sample_id
        if not well_mask.any():
            continue
        well_annot = annot_df.loc[annot_df["suffix"] == suffix].set_index("bc_seq")["predicted.celltype.l2"]
        # Extract raw barcode: strip {sample_id}- prefix, then strip -1 suffix
        raw_bcs = pd.Series(
            [
                name[len(sid) + 1 :].rsplit("-", 1)[0]
                for name, sid in zip(adata.obs_names[well_mask], adata.obs.loc[well_mask, "sample_id"], strict=True)
            ],
            index=adata.obs_names[well_mask],
        )
        annotations.loc[well_mask] = raw_bcs.map(well_annot).values
    n_annotated = annotations.notna().sum()
    print(f"TEA-seq: annotated {n_annotated}/{adata.n_obs} cells from Figure4 CSV")

    # --- UUID barcode mapping ---
    # Fragment files use UUID barcodes, not cellranger barcodes.
    # Convert obs_names from {sample_id}-{cellranger_bc} to {sample_id}-{uuid_bc}
    # using metadata CSVs that provide the mapping.
    import glob as _glob

    tea_seq_base = data_folder.rstrip("/")
    bc_map = {}  # cellranger_barcode -> uuid_barcode
    for meta_file in sorted(
        _glob.glob(f"{tea_seq_base}/tea_seq/GSM*/GSM*_metadata.csv.gz")
        + _glob.glob(f"{tea_seq_base}/multiome/GSM*/GSM*_metadata.csv.gz")
    ):
        meta_df = pd.read_csv(meta_file, usecols=["barcodes", "original_barcodes"])
        for _, row in meta_df.iterrows():
            bc_map[row["original_barcodes"]] = row["barcodes"]
    print(f"TEA-seq UUID barcode mapping: {len(bc_map)} entries")

    # Remap obs_names: replace cellranger barcode with UUID
    new_names = []
    n_uuid_mapped = 0
    for name, sid in zip(adata.obs_names, adata.obs["sample_id"], strict=True):
        raw_bc = name[len(sid) + 1 :]
        if not raw_bc.endswith("-1"):
            raw_bc = raw_bc + "-1"
        uuid_bc = bc_map.get(raw_bc)
        if uuid_bc is not None:
            new_names.append(f"{sid}-{uuid_bc}")
            n_uuid_mapped += 1
        else:
            new_names.append(name)
    adata.obs_names = new_names
    # Re-index annotations to match new UUID obs_names
    annotations.index = adata.obs_names
    print(f"  UUID mapped: {n_uuid_mapped}/{adata.n_obs}")

    # Set obs columns
    adata.obs["batch"] = adata.obs["sample_id"].values
    adata.obs["site"] = "allen_institute"
    adata.obs["donor"] = "pbmc_donor1"
    adata.obs["dataset"] = "pbmc_tea_seq"
    adata.obs["tissue"] = "pbmc"
    adata.obs["condition"] = "healthy"
    adata.obs["age_group"] = "adult"
    adata.obs["sex"] = "unknown"

    # original_annotation: available for GSM5123951-54 (from Figure4 CSV)
    adata.obs["original_annotation"] = annotations.values

    # harmonized_annotation via TEA_SEQ_MAP
    harm_map = _get_harmonization_maps().get("pbmc_tea_seq", {})
    adata.obs["harmonized_annotation"] = adata.obs["original_annotation"].map(harm_map)

    # Clean up temporary column
    adata.obs.drop(columns=["sample_id"], inplace=True)

    return _finalize(adata)


def load_covid_pbmc_gse239799(
    data_folder: str = "/nfs/team283/vk7/sanger_projects/large_data/multiome_pbmc/GSE239799/",
) -> sc.AnnData:
    """Load COVID infant PBMC multiome dataset GSE239799 (GEX only).

    43 samples from 18 subjects (longitudinal). No cell annotations available.

    Returns
    -------
    Standardized AnnData with raw GEX counts.
    """
    csv_path = os.path.join(data_folder, "sample_mapping.csv")
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


def load_infant_adult_spleen_gse311423(
    data_folder: str = "/nfs/team283/vk7/sanger_projects/large_data/multiome_spleen_lung/GSE311423/",
) -> sc.AnnData:
    """Load infant/adult spleen multiome data from GSE311423 (GEX only).

    Loads 5 samples (3 infant, 2 adult) from the sample mapping CSV,
    filters to Gene Expression features, sets ENSEMBL IDs as var_names,
    and standardizes obs.

    There are no cell annotations for this dataset.

    Returns
    -------
    Standardized AnnData with raw GEX counts.
    """
    sample_csv = os.path.join(data_folder, "sample_mapping.csv")
    sample_df = pd.read_csv(sample_csv)

    adatas = []
    for _, row in sample_df.iterrows():
        adata_i = sc.read_10x_h5(row["gex_h5_path"])

        # Keep only Gene Expression features
        adata_i = adata_i[:, adata_i.var["feature_types"] == "Gene Expression"].copy()

        # ENSEMBL IDs as var_names, gene symbols in var["SYMBOL"]
        adata_i = _set_ensembl_var_names(adata_i)

        # Prefix obs_names with library_id: {library_id}-{barcode} (cell2state format)
        library_id = row["library_id"]
        adata_i.obs_names = [f"{library_id}-{bc}" for bc in adata_i.obs_names]

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


def load_crohns_pbmc_gse244831(
    data_folder: str = "/nfs/team283/vk7/sanger_projects/large_data/multiome_pbmc/GSE244831/",
) -> sc.AnnData:
    """Load Crohn's disease PBMC multiome dataset (GEX only) from GSE244831.

    Loads 13 samples from 10x MTX files, joins cell-level annotations, and
    standardizes obs columns.

    Returns
    -------
    Standardized AnnData with raw GEX counts.
    """
    base = data_folder.rstrip("/")
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

    # Load cell annotations (safe reindex, never use .obs.join())
    annot = pd.read_csv(
        os.path.join(base, "annotations", "GSE244831_cell_annotations.csv"),
    )
    # Annotation barcodes may use # separator (e.g. Pool_8#GTTAACGGTGCTTTAC-1)
    # but adata obs_names now use - separator — convert to match
    annot.index = annot["barcode"].str.replace("#", "-", n=1).values
    annot = annot.loc[annot.index.isin(adata.obs_names)]
    annot_reindexed = annot.reindex(index=adata.obs_names)
    for col in annot.columns:
        adata.obs[col] = annot_reindexed[col].values
    n_matched = annot_reindexed.iloc[:, 0].notna().sum()
    print(f"  Annotations matched: {n_matched}/{adata.n_obs} cells")

    # Build fragment file path lookup (sample_id -> path)
    frag_map = dict(zip(sample_mapping["sample_id"], sample_mapping["fragment_file_path"], strict=True))

    # Set standardized obs columns
    # Use "sample" (from _read_10x_mtx, always populated) not "Sample" (from
    # annotation CSV, NaN for ~28% unmatched cells).
    adata.obs["batch"] = adata.obs["sample"].values
    adata.obs["site"] = "emory"
    # Donors_IDs from annotation; fall back to sample for unannotated cells
    adata.obs["donor"] = adata.obs["Donors_IDs"].fillna(adata.obs["sample"]).values
    adata.obs["dataset"] = "crohns_pbmc"
    adata.obs["tissue"] = "pbmc"
    adata.obs["condition"] = adata.obs["Status"].str.lower().fillna("unknown").values
    adata.obs["age_group"] = "adult"
    adata.obs["sex"] = adata.obs["Sex"].fillna("unknown").values
    adata.obs["original_annotation"] = adata.obs["Celltypes"].values
    adata.obs["harmonized_annotation"] = adata.obs["Celltypes"].map(_get_harmonization_maps()["crohns_pbmc"]).values
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


def load_lung_spleen_gse319044(
    data_folder: str = "/nfs/team283/vk7/sanger_projects/large_data/multiome_spleen_lung/GSE319044/",
) -> sc.AnnData:
    """Load lung and spleen immune multiome data (GEX only) from GSE319044.

    Loads 16 samples (9 lung + 7 spleen) from 10x multiome experiments,
    reads GEX counts from MTX files, and joins with cell-type annotations.

    Returns
    -------
    Standardized AnnData with raw GEX counts.
    """
    # --- 1. Read sample mapping ---
    sample_mapping_path = os.path.join(data_folder, "sample_mapping.csv")
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
    annotation_path = os.path.join(data_folder, "series_level/GSE319044_snRNA_cluster_labels.csv.gz")
    ann = pd.read_csv(annotation_path, index_col=0)

    # Annotation barcodes have Seurat-style suffixes (e.g. AAACAGCCAACAACAA-2_1).
    # The 10x barcodes in adata are {sample_id}-{barcode} where barcode is the raw
    # 10x barcode (e.g. AAACAGCCAACAACAA-1). We match via:
    #   1. Extract the 16-mer nucleotide sequence from the annotation barcode
    #   2. Use annotation library_id + tissue.ident to find the sample_id
    #   3. Reconstruct as {sample_id}-{nucleotide_seq}-1

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
            new_index.append(f"{sample_id}-{nuc}-1")
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
    adata.obs.loc[common_cells, "harmonized_annotation"] = (
        ann_matched["CellType"].map(_get_harmonization_maps()["lung_spleen_gse319044"]).values
    )

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


# ---------------------------------------------------------------------------
# ATAC barcode utilities
# ---------------------------------------------------------------------------


def rename_obs_for_cell2state(adata: sc.AnnData) -> sc.AnnData:
    """DEPRECATED: obs_names are now in cell2state format from each loader.

    This function is kept as a no-op assertion that format is correct.
    Each dataset loader now produces ``{batch}-{barcode}`` obs_names directly,
    and TEA-seq UUID mapping is done inside ``load_tea_seq_pbmc()``.
    """
    import warnings

    warnings.warn(
        "rename_obs_for_cell2state() is deprecated — obs_names are now in "
        "{batch}-{barcode} format from each loader. This is a no-op.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Verify format: all obs_names should contain at least one hyphen
    bad = [n for n in adata.obs_names[:100] if "-" not in n]
    if bad:
        raise ValueError(f"obs_names not in {{batch}}-{{barcode}} format. Examples: {bad[:3]}")
    return adata
