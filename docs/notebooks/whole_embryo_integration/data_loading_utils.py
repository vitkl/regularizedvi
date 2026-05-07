"""HDMA (Human Development Multiomic Atlas) loaders.

Builds AnnData from HDMA per-sample MTX RNA matrices and joins per-cell
metadata from the global `per_cell_meta.csv` deposit. Fragment file paths
are carried through `obs.fragment_file_path` so downstream ATAC steps
(SnapATAC2 etc.) can resolve them per cell.

Layout assumption (set up by scripts/zenodo_download/download_hdma.py):

    <data_root>/
      manifest/hdma_sample_mapping.csv
      annotations/per_cell_meta.csv
      rna/<sample_id>/{barcodes,features,matrix}.{tsv,mtx}.gz
      atac/<sample_id>/fragments.tsv.gz{,.tbi}

Cell barcodes use ArchR-style keys `<sample_id>#<barcode>` to match
`per_cell_meta.csv`. HDMA features.tsv.gz has 2 columns (Ensembl ID,
gene symbol) — different from 10x Cell Ranger's 3-column format.
"""

from __future__ import annotations

import gc
import os

import anndata
import pandas as pd
import scanpy as sc
import scipy.io

HDMA_ORGANS = (
    "Adrenal",
    "Brain",
    "Eye",
    "Heart",
    "Liver",
    "Lung",
    "Muscle",
    "Skin",
    "Spleen",
    "StomachEsophagus",
    "Thymus",
    "Thyroid",
)


def _read_hdma_mtx(barcodes_path: str, features_path: str, matrix_path: str, sample_id: str) -> sc.AnnData:
    """Read one HDMA RNA sample (2-column features, ArchR-style barcode keys)."""
    mat = scipy.io.mmread(matrix_path).T.tocsr()
    barcodes = pd.read_csv(barcodes_path, header=None, sep="\t")[0].values
    features = pd.read_csv(features_path, header=None, sep="\t")

    adata = sc.AnnData(
        X=mat,
        obs=pd.DataFrame(index=barcodes),
        var=pd.DataFrame(index=features[0].values),
    )
    if features.shape[1] >= 2:
        adata.var["SYMBOL"] = features[1].values
    adata.var_names_make_unique()

    adata.obs_names = [f"{sample_id}#{bc}" for bc in adata.obs_names]
    adata.obs["sample_id"] = sample_id
    return adata


def load_hdma_organ(
    organ: str,
    data_root: str = "/nemo/lab/briscoej/home/users/kleshcv/large_data/HDMA",
    join_metadata: bool = True,
) -> sc.AnnData:
    """Load all samples for one HDMA organ into a single AnnData.

    Parameters
    ----------
    organ
        One of HDMA_ORGANS (case-sensitive, e.g. "Brain", "StomachEsophagus").
    data_root
        Root populated by download_hdma.py (contains manifest/, rna/, atac/,
        annotations/).
    join_metadata
        If True, join per-cell annotations from annotations/per_cell_meta.csv
        and drop cells absent from that table (these are the cells the paper
        excluded during QC).
    """
    if organ not in HDMA_ORGANS:
        raise ValueError(f"Unknown organ {organ!r}; expected one of {HDMA_ORGANS}")

    mapping_path = os.path.join(data_root, "manifest", "hdma_sample_mapping.csv")
    sample_df = pd.read_csv(mapping_path)
    sample_df = sample_df[sample_df["organ"] == organ].reset_index(drop=True)
    if sample_df.empty:
        raise RuntimeError(f"No samples for organ={organ!r} in {mapping_path}")

    adatas = []
    for _, row in sample_df.iterrows():
        ad = _read_hdma_mtx(
            row["barcodes_path"],
            row["features_path"],
            row["matrix_path"],
            row["sample_id"],
        )
        print(f"  {row['sample_id']}: {ad.n_obs} cells x {ad.n_vars} genes")
        adatas.append(ad)
    adata = anndata.concat(adatas, join="outer")
    del adatas
    gc.collect()
    print(f"  concat ({organ}): {adata.n_obs} cells x {adata.n_vars} genes")

    frag_map = dict(zip(sample_df["sample_id"], sample_df["fragment_file_path"], strict=True))
    adata.obs["fragment_file_path"] = [frag_map.get(s, "") for s in adata.obs["sample_id"].values]

    if join_metadata:
        meta_path = os.path.join(data_root, "annotations", "per_cell_meta.csv")
        meta = pd.read_csv(meta_path, index_col=0)
        before = adata.n_obs
        common = adata.obs_names.intersection(meta.index)
        adata = adata[common].copy()
        meta_aligned = meta.reindex(index=adata.obs_names)
        for col in meta_aligned.columns:
            adata.obs[col] = meta_aligned[col].values
        print(f"  metadata join: {adata.n_obs}/{before} cells matched per_cell_meta.csv")

    return adata


def load_hdma_all(
    data_root: str = "/nemo/lab/briscoej/home/users/kleshcv/large_data/HDMA",
    organs: tuple[str, ...] = HDMA_ORGANS,
    join_metadata: bool = True,
) -> sc.AnnData:
    """Concatenate all HDMA organs into a single AnnData."""
    parts = []
    for organ in organs:
        print(f"[{organ}]")
        ad = load_hdma_organ(organ, data_root=data_root, join_metadata=join_metadata)
        ad.obs["organ"] = organ
        parts.append(ad)
    adata = anndata.concat(parts, join="outer")
    print(f"[all]: {adata.n_obs} cells x {adata.n_vars} genes")
    return adata
