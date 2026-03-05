"""Build multimodal MuData from source h5ad + RDS files with integrated cell QC filtering.

Merges the NB1b (build) and NB1d (cell filter) pipeline steps into a single script.
Key optimization: applies cell QC filters on the RNA h5ad BEFORE loading RDS files,
then subsets each per-embryo RDS to passing cells + shared genes immediately.

Usage from NB2:
    from build_gastrula_mudata import build_gastrula_mudata
    mdata = build_gastrula_mudata(DATA_DIR)

Standalone:
    python build_gastrula_mudata.py --data-dir /path/to/data
"""

from __future__ import annotations

# matplotlib/rds2py conflict workaround — must init matplotlib BEFORE rds2py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_fig = plt.figure()
plt.close(_fig)

import argparse  # noqa: E402
import gc  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import anndata as ad  # noqa: E402
import mudata as mu  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
import scanpy as sc  # noqa: E402
from rds2py import read_rds  # noqa: E402

from regularizedvi.utils import filter_genes  # noqa: E402

_DEFAULT_GENE_FILTER_KWARGS = {
    "cell_count_cutoff": 30,
    "cell_percentage_cutoff2": 0.005,
    "nonz_mean_cutoff": 1.05,
}


def _rss_gb():
    return psutil.Process().memory_info().rss / 1e9


def _print(msg, verbose):
    if verbose:
        print(msg)


def _load_rds_modality(
    rds_dir: str,
    prefix: str,
    n_embryos: int,
    passing_cells: set,
    shared_genes: pd.Index,
    rds_gene_index: pd.Index,
    col_indices: np.ndarray,
    gene_filter_kwargs: dict,
    verbose: bool = True,
) -> ad.AnnData:
    """Load per-embryo RDS files, subset to passing cells + shared genes, concat, filter genes."""
    adatas = []
    t0 = time.time()
    for i in range(1, n_embryos + 1):
        if verbose and i % 10 == 0:
            _print(f"  {prefix} embryo {i}/{n_embryos}, RSS={_rss_gb():.1f}GB", verbose)
        obj = read_rds(os.path.join(rds_dir, f"embryo_{i}_{prefix}.rds"))
        all_cells = list(obj.dimnames[1])

        # Subset to QC-passing cells
        cell_mask = np.array([c in passing_cells for c in all_cells])
        if not cell_mask.any():
            del obj
            gc.collect()
            continue

        mat = obj.matrix.T.tocsr().astype(np.uint16)
        del obj

        # Subset rows (cells) and columns (genes to shared_genes)
        mat = mat[cell_mask][:, col_indices]
        kept_cells = [c for c, m in zip(all_cells, cell_mask, strict=True) if m]

        adatas.append(
            ad.AnnData(
                X=mat,
                obs=pd.DataFrame(index=kept_cells),
                var=pd.DataFrame(index=shared_genes),
            )
        )
        del mat
        gc.collect()

    elapsed = time.time() - t0
    _print(f"Loaded {n_embryos} {prefix} files in {elapsed:.0f}s", verbose)

    adata = ad.concat(adatas, join="inner")
    del adatas
    gc.collect()
    _print(f"  Concat: {adata.shape}, dtype={adata.X.dtype}, RSS={_rss_gb():.1f}GB", verbose)

    # Gene filtering
    selected = filter_genes(adata, **gene_filter_kwargs)
    adata_filt = adata[:, selected].copy()
    _print(f"  Gene filter: {adata.n_vars} -> {adata_filt.n_vars} genes", verbose)
    del adata
    gc.collect()
    _print(f"  RSS after filter: {_rss_gb():.1f}GB", verbose)

    return adata_filt


def _plot_qc(obs_prefilter: pd.DataFrame, cell_mask: np.ndarray, save_path: str, thresholds: dict):
    """Save 2x2 QC plot with filter thresholds."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # n_genes_by_counts
    ax = axes[0, 0]
    ax.hist(obs_prefilter["n_genes_by_counts"].values, bins=100, color="steelblue", alpha=0.7)
    ax.axvline(thresholds["min_genes"], color="red", ls="--", label=f">{thresholds['min_genes']}")
    ax.set_xlabel("n_genes_by_counts")
    ax.set_ylabel("Cells")
    ax.legend()

    # total_counts
    ax = axes[0, 1]
    ax.hist(obs_prefilter["total_counts"].values, bins=100, color="steelblue", alpha=0.7)
    ax.axvline(thresholds["min_counts"], color="red", ls="--", label=f">{thresholds['min_counts']}")
    ax.axvline(thresholds["max_counts"], color="orange", ls="--", label=f"<{thresholds['max_counts']}")
    ax.set_xlabel("total_counts")
    ax.set_ylabel("Cells")
    ax.legend()

    # mt_frac
    ax = axes[1, 0]
    ax.hist(obs_prefilter["mt_frac"].values, bins=100, color="steelblue", alpha=0.7)
    ax.axvline(thresholds["max_mt_frac"], color="red", ls="--", label=f"<{thresholds['max_mt_frac']}")
    ax.set_xlabel("mt_frac")
    ax.set_ylabel("Cells")
    ax.legend()

    # Scatter: total_counts vs n_genes, colored by pass/fail
    ax = axes[1, 1]
    fail = ~cell_mask
    if fail.any():
        ax.scatter(
            obs_prefilter.loc[fail, "total_counts"],
            obs_prefilter.loc[fail, "n_genes_by_counts"],
            s=0.1,
            alpha=0.1,
            c="red",
            label="filtered",
            rasterized=True,
        )
    ax.scatter(
        obs_prefilter.loc[cell_mask, "total_counts"],
        obs_prefilter.loc[cell_mask, "n_genes_by_counts"],
        s=0.1,
        alpha=0.1,
        c="steelblue",
        label="kept",
        rasterized=True,
    )
    ax.set_xlabel("total_counts")
    ax.set_ylabel("n_genes_by_counts")
    ax.legend(markerscale=20)

    n_pass = cell_mask.sum()
    n_total = len(cell_mask)
    fig.suptitle(f"Cell QC: {n_pass:,} / {n_total:,} cells kept ({100 * n_pass / n_total:.1f}%)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"QC plot saved: {save_path}")


def build_gastrula_mudata(
    data_dir: str,
    *,
    min_genes: int = 800,
    min_counts: int = 1000,
    max_counts: int = 50000,
    max_mt_frac: float = 0.10,
    gene_filter_kwargs: dict | None = None,
    n_embryos: int = 74,
    rds_subdir: str = "embryo_exon_intron",
    save_h5mu: str | None = None,
    save_qc_plot: str | None = None,
    verbose: bool = True,
) -> mu.MuData:
    """Build 3-modality MuData (rna, spliced, unspliced) from source files with cell QC.

    Parameters
    ----------
    data_dir
        Directory containing gastrula_to_pup_rna_qc.h5ad and embryo_exon_intron/ subdirectory.
    min_genes
        Minimum n_genes_by_counts per cell.
    min_counts
        Minimum total_counts per cell.
    max_counts
        Maximum total_counts per cell.
    max_mt_frac
        Maximum mitochondrial fraction per cell.
    gene_filter_kwargs
        Kwargs passed to regularizedvi.utils.filter_genes(). Defaults to
        cell_count_cutoff=30, cell_percentage_cutoff2=0.005, nonz_mean_cutoff=1.05.
    n_embryos
        Number of embryos (1..n_embryos).
    rds_subdir
        Subdirectory within data_dir containing RDS files.
    save_h5mu
        If provided, save the MuData to this path.
    save_qc_plot
        If provided, save QC filter plot to this path.
    verbose
        Print progress messages.

    Returns
    -------
    MuData with modalities "rna", "spliced", "unspliced".
    """
    if gene_filter_kwargs is None:
        gene_filter_kwargs = _DEFAULT_GENE_FILTER_KWARGS.copy()
    rds_dir = os.path.join(data_dir, rds_subdir)
    t_start = time.time()

    # --- Step 1: Load RNA h5ad ---
    h5ad_path = os.path.join(data_dir, "gastrula_to_pup_rna_qc.h5ad")
    _print(f"Loading {h5ad_path}...", verbose)
    adata = sc.read_h5ad(h5ad_path)
    _print(f"RNA: {adata.shape}, dtype={adata.X.dtype}, RSS={_rss_gb():.1f}GB", verbose)

    # --- Step 2: Load scrublet scores (auto-detect) ---
    scrublet_csv = os.path.join(data_dir, "scrublet_scores.csv")
    if os.path.exists(scrublet_csv):
        scores = pd.read_csv(scrublet_csv, index_col="cell_id")
        adata.obs["doublet_score"] = scores["doublet_score"].reindex(adata.obs_names).values
        adata.obs["predicted_doublet"] = scores["predicted_doublet"].reindex(adata.obs_names).fillna(False).values
        n_scored = scores.index.isin(adata.obs_names).sum()
        _print(f"Scrublet scores: {n_scored:,} / {adata.n_obs:,} cells", verbose)
        del scores
    else:
        _print(f"Scrublet scores not found at {scrublet_csv} — skipping", verbose)

    # --- Step 3: Cell QC filter (EARLY, before loading RDS) ---
    obs = adata.obs
    cell_mask = (
        (obs["n_genes_by_counts"] > min_genes)
        & (obs["total_counts"] > min_counts)
        & (obs["total_counts"] < max_counts)
        & (obs["mt_frac"] < max_mt_frac)
    )
    n_before = adata.n_obs
    n_after = cell_mask.sum()
    _print(
        f"Cell QC: {n_before:,} -> {n_after:,} cells "
        f"({n_before - n_after:,} removed, {100 * (n_before - n_after) / n_before:.1f}%)",
        verbose,
    )

    # --- Step 4: QC plot (before filtering, to show distributions) ---
    if save_qc_plot:
        obs_prefilter = obs[["n_genes_by_counts", "total_counts", "mt_frac"]].copy()
        _plot_qc(
            obs_prefilter,
            cell_mask.values,
            save_qc_plot,
            {"min_genes": min_genes, "min_counts": min_counts, "max_counts": max_counts, "max_mt_frac": max_mt_frac},
        )
        del obs_prefilter

    # Save passing cell IDs and obs BEFORE subsetting (to avoid double-memory peak)
    passing_idx = cell_mask.values
    passing_cell_names = adata.obs_names[passing_idx]
    passing_cells = set(passing_cell_names)
    shared_obs = adata.obs.loc[passing_idx].copy()
    del cell_mask
    _print(f"RSS before freeing full RNA: {_rss_gb():.1f}GB", verbose)

    # --- Step 5: Shared genes + RNA gene filtering ---
    # Get shared genes BEFORE freeing adata (need var_names)
    _print("Getting shared genes from first RDS file...", verbose)
    _exon = read_rds(os.path.join(rds_dir, "embryo_1_exp_exon.rds"))
    rds_gene_index = pd.Index(list(_exon.dimnames[0]))
    del _exon
    gc.collect()

    shared_genes = adata.var_names.intersection(rds_gene_index)
    _print(f"Shared genes: {len(shared_genes)} / {adata.n_vars} (RNA) / {len(rds_gene_index)} (RDS)", verbose)

    # Precompute column indices for RDS → shared_genes mapping
    col_indices = rds_gene_index.get_indexer(shared_genes)
    assert (col_indices >= 0).all(), "Some shared genes not found in RDS gene index"

    # Subset RNA to passing cells + shared genes, then gene filter
    # Use direct indexing on the view to avoid full copy of original
    adata_sub = adata[passing_idx][:, shared_genes]
    n_vars_orig = adata.n_vars
    del adata  # free the 188GB original BEFORE copying the subset
    gc.collect()
    _print(f"RSS after freeing original RNA: {_rss_gb():.1f}GB", verbose)

    adata_sub = adata_sub.copy()  # materialize the view (now only ~120GB)
    gc.collect()

    selected_rna = filter_genes(adata_sub, **gene_filter_kwargs)
    adata_rna = adata_sub[:, selected_rna].copy()
    _print(f"RNA genes: {n_vars_orig} -> {len(shared_genes)} (shared) -> {adata_rna.n_vars} (filtered)", verbose)

    rna_cells = pd.Index(passing_cell_names)
    del adata_sub
    gc.collect()
    _print(f"RSS after RNA gene filter: {_rss_gb():.1f}GB", verbose)

    # --- Step 6: Load exon (spliced) ---
    _print("\nLoading exon (spliced) RDS files...", verbose)
    adata_spliced = _load_rds_modality(
        rds_dir,
        "exp_exon",
        n_embryos,
        passing_cells,
        shared_genes,
        rds_gene_index,
        col_indices,
        gene_filter_kwargs,
        verbose,
    )

    # --- Step 7: Load intron (unspliced) ---
    _print("\nLoading intron (unspliced) RDS files...", verbose)
    adata_unspliced = _load_rds_modality(
        rds_dir,
        "exp_intron",
        n_embryos,
        passing_cells,
        shared_genes,
        rds_gene_index,
        col_indices,
        gene_filter_kwargs,
        verbose,
    )

    # --- Step 8: Build MuData ---
    # Align all modalities to shared cells
    shared_cells = rna_cells.intersection(adata_spliced.obs_names).intersection(adata_unspliced.obs_names)
    _print(f"\nShared cells: {len(shared_cells):,}", verbose)

    adata_rna = adata_rna[shared_cells].copy()
    adata_spliced = adata_spliced[shared_cells].copy()
    adata_unspliced = adata_unspliced[shared_cells].copy()

    obs_aligned = shared_obs.reindex(shared_cells).copy()
    adata_rna.obs = obs_aligned.copy()
    adata_spliced.obs = obs_aligned.copy()
    adata_unspliced.obs = obs_aligned.copy()
    del shared_obs, obs_aligned

    mdata = mu.MuData({"rna": adata_rna, "spliced": adata_spliced, "unspliced": adata_unspliced})
    del adata_rna, adata_spliced, adata_unspliced
    gc.collect()

    _print(f"\nMuData: {mdata.n_obs:,} cells × {mdata.n_vars:,} vars", verbose)
    for mod in mdata.mod:
        _print(f"  {mod}: {mdata[mod].shape}, dtype={mdata[mod].X.dtype}", verbose)
    _print(f"RSS final: {_rss_gb():.1f}GB", verbose)
    _print(f"Total time: {(time.time() - t_start) / 60:.1f} min", verbose)

    # --- Step 9: Optional save ---
    if save_h5mu:
        _print(f"\nSaving to {save_h5mu}...", verbose)
        t_save = time.time()
        mdata.write_h5mu(save_h5mu)
        _print(f"Saved ({os.path.getsize(save_h5mu) / 1e9:.1f} GB) in {(time.time() - t_save) / 60:.1f} min", verbose)

    return mdata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build gastrula-to-pup multimodal MuData")
    parser.add_argument("--data-dir", required=True, help="Directory with h5ad and RDS files")
    parser.add_argument("--min-genes", type=int, default=800)
    parser.add_argument("--min-counts", type=int, default=1000)
    parser.add_argument("--max-counts", type=int, default=50000)
    parser.add_argument("--max-mt-frac", type=float, default=0.10)
    parser.add_argument("--n-embryos", type=int, default=74)
    parser.add_argument("--save-h5mu", default=None, help="Path to save output h5mu")
    parser.add_argument("--save-qc-plot", default=None, help="Path to save QC plot PNG")
    args = parser.parse_args()

    mdata = build_gastrula_mudata(
        args.data_dir,
        min_genes=args.min_genes,
        min_counts=args.min_counts,
        max_counts=args.max_counts,
        max_mt_frac=args.max_mt_frac,
        n_embryos=args.n_embryos,
        save_h5mu=args.save_h5mu,
        save_qc_plot=args.save_qc_plot,
    )
    print(f"\nDone. MuData: {mdata.n_obs:,} cells × {mdata.n_vars:,} vars")
