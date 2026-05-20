"""Inspect per-dataset metadata for ALL 11 v2 datasets; emit column-rename diff blocks.

Phase 0 step 0.6 of immune_integration_v2 (see [plan](../../.claude/plans/implement-these-steps-in-tranquil-parasol.md)).

For each v2 dataset, reads the metadata source (sample_mapping.csv + per-sample obs / SDRF /
Zenodo TXT / h5ad obs) and emits a 5-column markdown diff block proposing how each
source column maps to v1 STANDARD_OBS_COLS:

    | Source column | Example values | n unique | Proposed v1 obs column | Notes |

The user reviews + edits + copies approved rows into canonical
`docs/notebooks/immune_integration_v2/metadata_harmonization.md`.

v1 STANDARD_OBS_COLS (per data_loading_utils.py L30-L46):
    batch, site, donor, dataset, tissue, condition, age_group, sex,
    original_annotation, harmonized_annotation,
    level_1, level_2, level_3, level_4, fragment_file_path
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import h5py
import pandas as pd

DATA_ROOT = Path("/nemo/lab/briscoej/home/users/kleshcv/large_data")
DEFAULT_OUT = Path(
    "/nemo/lab/briscoej/home/users/kleshcv/my_packages/regularizedvi/"
    "docs/notebooks/immune_integration_v2/metadata_harmonization_proposed.md"
)

DATASETS = (
    "htan_pan_cancer",
    "gbm_space",
    "hippocampus_aging",
    "lung_smoking",
    "intestine_hickey",
    "hdma_immune",
    "ad_brain_3region",
    "bach2_ap1_gut_tcells",
    "bcg_trained_immunity",
    "rorgt_dc_tonsil",
    "down_fetal_blood",
)

V1_STANDARD_OBS_COLS = (
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
    # v2 extension (parent plan Open Issue #7):
    "cancer_type",
)


# ---------------------------------------------------------------------------
# Per-dataset configs: loader callable + proposed source → v1 column dict
# ---------------------------------------------------------------------------


def _h5py_categorical(node) -> list[str]:
    """Decode an h5py categorical node (categories + codes) to a list of strings."""
    cats = [c.decode() if isinstance(c, bytes) else c for c in node["categories"][:]]
    codes = node["codes"][:]
    return [cats[c] if c >= 0 else "<NA>" for c in codes]


def _read_h5ad_obs_h5py(h5ad: Path) -> pd.DataFrame:
    """Read all obs columns from an h5ad via h5py (handles categorical + raw dtypes)."""
    cols: dict[str, list] = {}
    with h5py.File(h5ad, "r") as f:
        for key in f["obs"].keys():
            node = f["obs"][key]
            if isinstance(node, h5py.Group) and "categories" in node and "codes" in node:
                cols[key] = _h5py_categorical(node)
            elif isinstance(node, h5py.Dataset):
                arr = node[:]
                if arr.dtype.kind in ("S", "O"):
                    cols[key] = [v.decode() if isinstance(v, bytes) else v for v in arr]
                else:
                    cols[key] = arr.tolist()
        # Align lengths (skip unequal-length nodes like /obs/__categories)
        n = max((len(v) for v in cols.values()), default=0)
        cols = {k: v for k, v in cols.items() if len(v) == n}
    return pd.DataFrame(cols)


def load_htan_pan_cancer() -> pd.DataFrame:
    """Read HTAN sample_mapping.csv and inner-join Ding-lab Sample_ID lookup on piece_id."""
    sm = pd.read_csv(DATA_ROOT / "pan_cancer_multiome/sample_mapping.csv")
    lookup_path = DATA_ROOT / "pan_cancer_multiome/annotations/pan_cancer_multiome_sample_lookup.csv"
    if lookup_path.is_file():
        ld = pd.read_csv(lookup_path)
        # sm.sample_id matches ld.piece_id (verified: both 'CE336E1-S1', 'BM2', etc.)
        if "piece_id" in ld.columns and "sample_id" in sm.columns:
            n_before = len(sm)
            sm = sm.merge(ld, left_on="sample_id", right_on="piece_id", how="left", suffixes=("", "_lookup"))
            n_matched = sm["piece_id"].notna().sum()
            print(
                f"  HTAN sample_mapping ⨝ Ding-lab lookup: {n_matched}/{n_before} rows matched",
                file=sys.stderr,
            )
        else:
            print("  WARN: cannot join — missing piece_id or sample_id columns", file=sys.stderr)
    return sm


HTAN_PROPOSED = {
    "sample_id": "batch",
    "cancer_type": "cancer_type",
    "organ": "tissue",
    "diagnosis": "condition",
    "fragment_file_path": "fragment_file_path",
    # Ding-lab lookup join (post-fix; matches lowercase snake_case column names):
    "piece_id": "<auxiliary; join key only>",
    "donor_id": "donor",
    "biospecimen_id": "<auxiliary; HTAN biospecimen ID>",
    "geo_sample_name": "<auxiliary; drop>",
    "atac_data_type": "<auxiliary; drop>",
    "raw_data_uploaded_to": "<auxiliary; drop>",
    "processed_data_uploaded_to": "<auxiliary; drop>",
    "cds_sample_name": "<auxiliary; drop>",
    "gdc_bam_file_id": "<auxiliary; drop>",
    "source_sheet": "<auxiliary; drop>",
}


def load_gbm_space() -> pd.DataFrame:
    """Read GBM sample_mapping.csv and append h5ad obs unique-count placeholders."""
    sm = pd.read_csv(DATA_ROOT / "gbm/sample_mapping.csv")
    # Also pull a small sample of h5ad obs categoricals (donor_id, sample, site_id)
    h5ad = DATA_ROOT / "gbm/GBM_space_snRNA.h5ad"
    if h5ad.is_file():
        with h5py.File(h5ad, "r") as f:
            for key in ("donor_id", "sample", "site_id"):
                if key in f["obs"]:
                    vals = _h5py_categorical(f["obs"][key])
                    sm[f"_obs_{key}_unique"] = pd.NA  # placeholder; we surface unique counts in block
                    sm[f"_obs_{key}_n_unique"] = len(set(vals))
    return sm


GBM_PROPOSED = {
    "sample_dir": "batch",
    "gex_supplier_name": "<auxiliary; drop>",
    "atac_supplier_name": "<auxiliary; drop>",
    "gex_sanger_id": "<auxiliary; drop>",
    "atac_sanger_id": "<auxiliary; drop>",
    # h5ad-derived (loader will read these directly):
    "_obs_donor_id_n_unique": "donor",
    "_obs_sample_n_unique": "batch",
    "_obs_site_id_n_unique": "site",
}


def load_hippocampus_aging() -> pd.DataFrame:
    """Read hippocampus_aging sample_mapping + obs meta column placeholders."""
    sm = pd.read_csv(DATA_ROOT / "hippocampus_aging/sample_mapping.csv")
    tsv = (
        DATA_ROOT
        / "hippocampus_aging/annotations/GSE278576_hippocampus_RNA_seurat_object_filtered_cells_metadata.tsv.gz"
    )
    if tsv.is_file():
        meta_cols = pd.read_csv(tsv, sep="\t", nrows=0).columns.tolist()
        for c in meta_cols:
            if c not in sm.columns:
                sm[f"_meta_{c}"] = pd.NA
    return sm


HIPPOCAMPUS_PROPOSED = {
    "sample_id": "batch",
    "gse_id": "<auxiliary; drop>",
    "gsm_gex": "<auxiliary; drop>",
    "gsm_atac": "<auxiliary; drop>",
    "fragment_file_path": "fragment_file_path",
    "_meta_orig.ident": "donor",
    "_meta_Gender": "sex",
    "_meta_age": "age_group",
    "_meta_subclass": "original_annotation",
}


def load_lung_smoking() -> pd.DataFrame:
    """Read lung_smoking sample_mapping + Seurat meta.data column placeholders."""
    sm = pd.read_csv(DATA_ROOT / "lung_smoking/sample_mapping.csv")
    meta_csv = DATA_ROOT / "lung_smoking/annotations/lung_smoking_meta.csv"
    if meta_csv.is_file():
        meta = pd.read_csv(meta_csv, nrows=5, low_memory=False)
        for c in meta.columns:
            if c not in sm.columns:
                sm[f"_meta_{c}"] = pd.NA
    return sm


LUNG_SMOKING_PROPOSED = {
    "sample_id": "batch",
    "gse_id": "<auxiliary; drop>",
    "fragment_file_path": "fragment_file_path",
    "_meta_orig.ident": "donor",
    "_meta_smoker_status": "condition",
    "_meta_Sex": "sex",
    "_meta_Age": "age_group",
    "_meta_seurat_clusters": "<→ cell_type via SData4 (see annotation review)>",
}


def load_intestine_hickey() -> pd.DataFrame:
    """Read intestine_hickey sample_location_metadata.csv (Dryad ATAC location lookup)."""
    csv = DATA_ROOT / "intestine_hickey/annotations/sample_location_metadata.csv"
    if csv.is_file():
        return pd.read_csv(csv)
    return pd.DataFrame()


HICKEY_PROPOSED = {
    "SampleNameRNA": "batch",
    "SampleNameOnly": "<auxiliary; drop>",
    "Donor": "donor",
    "Multiome": "<filter: keep only 'Yes'>",
    "Location": "tissue",
}


def load_hdma_immune() -> pd.DataFrame:
    """Read HDMA hdma_sample_mapping.csv and restrict to SP/TM/LI organs."""
    sm = pd.read_csv(DATA_ROOT / "HDMA/manifest/hdma_sample_mapping.csv")
    sm = sm[sm["organ"].isin(["Spleen", "Thymus", "Liver"])]
    return sm


HDMA_PROPOSED = {
    "sample_id": "batch",
    "organ": "tissue",
    "donor_id": "donor",
    "batch": "<auxiliary; drop (HDMA's own batch ID, not v1 batch)>",
    "PCW": "age_group",
    "fragment_file_path": "fragment_file_path",
}


def load_ad_brain_3region() -> pd.DataFrame:
    """Read ad_brain_3region sample_mapping.csv."""
    return pd.read_csv(DATA_ROOT / "ad_brain_3region/sample_mapping.csv")


AD_BRAIN_PROPOSED = {
    "sample_id": "batch",
    "gse_id": "<auxiliary; drop>",
    "fragment_file_path": "fragment_file_path",
    # No per-cell author annotations → harmonized_annotation = NaN at loader.
}


def load_bach2_gut() -> pd.DataFrame:
    """Read bach2_ap1_gut_tcells sample_mapping.csv."""
    return pd.read_csv(DATA_ROOT / "bach2_ap1_gut_tcells/sample_mapping.csv")


BACH2_PROPOSED = {
    "sample_id": "batch",
    "gse_id": "<auxiliary; drop>",
    "fragment_file_path": "fragment_file_path",
    "barcodes_path": "<used by loader>",
    "features_path": "<used by loader>",
    "matrix_path": "<used by loader>",
    # Additional obsm['protein'] from HTO_ADT_of_* mtx; Zenodo TXT joined as obs cols
    # (hiv_status, tcr_clone, donor_demux) per parent plan Open Issue #6.
}


def load_bcg() -> pd.DataFrame:
    """Read bcg_trained_immunity sample_mapping + GSE h5ad obs column placeholders."""
    sm = pd.read_csv(DATA_ROOT / "bcg_trained_immunity/sample_mapping.csv")
    # Try to read h5ad obs columns
    for sub in ("rna/GSE295277/adata.h5ad", "rna/GSE295308/adata.h5ad"):
        h5ad = DATA_ROOT / "bcg_trained_immunity" / sub
        if h5ad.is_file():
            try:
                with h5py.File(h5ad, "r") as f:
                    for key in f["obs"].keys():
                        col = f"_obs_{key}"
                        if col not in sm.columns:
                            sm[col] = pd.NA
            except (OSError, KeyError) as exc:
                print(f"WARN: cannot read {h5ad} obs: {exc}", file=sys.stderr)
            break
    return sm


BCG_PROPOSED = {
    "sample_id": "batch",
    "gse_id": "<auxiliary; drop>",
    "fragment_file_path": "fragment_file_path",
    "_obs_barcode": "<row id; drop>",
    "_obs_orig_barcode": "<10x barcode; drop>",
    "_obs_sample": "batch",
    "_obs_experiment": "<auxiliary; record but drop>",
    "_obs_status": "condition",
    # Rename: dataset = "bcg_bladder_immunotherapy" (NOT bcg_trained_immunity) per parent plan §1 §5
}


def load_rorgt() -> pd.DataFrame:
    """Read rorgt_dc_tonsil sample_mapping.csv."""
    return pd.read_csv(DATA_ROOT / "rorgt_dc_tonsil/sample_mapping.csv")


RORGT_PROPOSED = {
    "sample_id": "batch",
    "gse_id": "<auxiliary; drop>",
    "fragment_file_path": "fragment_file_path",
    # Loader filter: drop rows with empty fragment_file_path (3 RNA-only Crohn's rows).
    # No per-cell author annotations → harmonized_annotation = NaN at loader.
}


def load_down_fetal_blood() -> pd.DataFrame:
    """Read down_fetal_blood E-MTAB-13070 SDRF metadata."""
    sdrf = DATA_ROOT / "down_fetal_blood/annotations/E-MTAB-13070.sdrf.txt"
    if not sdrf.is_file():
        return pd.DataFrame()
    sdrf_df = pd.read_csv(sdrf, sep="\t", low_memory=False)
    return sdrf_df


DOWN_FETAL_PROPOSED = {
    "Source Name": "batch",
    "Characteristics[individual]": "donor",
    "Characteristics[sex]": "sex",
    "Characteristics[age]": "age_group",
    "Characteristics[developmental stage]": "age_group",
    "Characteristics[organism part]": "tissue",
    "Characteristics[disease]": "condition",
    # CD45+ sorted → no immune filter at load; harmonized_annotation = NaN.
}


DATASET_CONFIG = {
    "htan_pan_cancer": (load_htan_pan_cancer, HTAN_PROPOSED, "HTAN pan-cancer"),
    "gbm_space": (load_gbm_space, GBM_PROPOSED, "GBM-Space"),
    "hippocampus_aging": (load_hippocampus_aging, HIPPOCAMPUS_PROPOSED, "hippocampus_aging"),
    "lung_smoking": (load_lung_smoking, LUNG_SMOKING_PROPOSED, "lung_smoking"),
    "intestine_hickey": (load_intestine_hickey, HICKEY_PROPOSED, "intestine_hickey"),
    "hdma_immune": (load_hdma_immune, HDMA_PROPOSED, "HDMA Spleen/Thymus/Liver"),
    "ad_brain_3region": (load_ad_brain_3region, AD_BRAIN_PROPOSED, "ad_brain_3region"),
    "bach2_ap1_gut_tcells": (load_bach2_gut, BACH2_PROPOSED, "bach2_ap1_gut_tcells"),
    "bcg_trained_immunity": (
        load_bcg,
        BCG_PROPOSED,
        "bcg_trained_immunity (→ rename dataset to bcg_bladder_immunotherapy)",
    ),
    "rorgt_dc_tonsil": (load_rorgt, RORGT_PROPOSED, "rorgt_dc_tonsil"),
    "down_fetal_blood": (load_down_fetal_blood, DOWN_FETAL_PROPOSED, "down_fetal_blood"),
}


# ---------------------------------------------------------------------------
# Diff-block writer
# ---------------------------------------------------------------------------


def _format_examples(vals: pd.Series, max_n: int = 5) -> str:
    """Return a short string of top-N value_counts as 'val'×count entries."""
    counts = vals.fillna("<NaN>").value_counts().head(max_n)
    return "; ".join(f"{str(k)[:30]!r}×{v}" for k, v in counts.items())


def write_metadata_block(
    out_path: Path,
    dataset_label: str,
    df: pd.DataFrame,
    proposed: dict[str, str],
) -> None:
    """Append a 5-column markdown block (Source col / Examples / n unique / Proposed / Notes) to out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = []
    lines.append(f"\n## {dataset_label} — column-rename proposal (inspect_dataset_metadata.py {timestamp})\n")
    lines.append(f"Source dataframe shape: {df.shape[0]:,} rows × {df.shape[1]} cols\n")
    lines.append("| Source column | Example values (top-5) | n unique | Proposed v1 obs column | Notes |")
    lines.append("|---|---|---:|---|---|")

    listed = set()
    for col in df.columns:
        prop = proposed.get(col, "<TODO: not in proposed_mapping>")
        n_unique = df[col].nunique(dropna=False)
        examples = _format_examples(df[col]) if not df.empty else "<empty>"
        note = ""
        if prop not in V1_STANDARD_OBS_COLS and not prop.startswith("<"):
            note = "NEW v1 column?"
        safe_col = str(col).replace("|", "\\|")
        safe_ex = examples.replace("|", "\\|")
        lines.append(f"| `{safe_col}` | {safe_ex} | {n_unique} | {prop} | {note} |")
        listed.add(col)

    extras = [k for k in proposed if k not in listed]
    if extras:
        lines.append(
            "\n_Proposed columns NOT present in source dataframe (placeholders for nested obs / Zenodo / SDRF joins):_\n"
        )
        for k in extras:
            lines.append(f"- `{k}` → `{proposed[k]}`")

    text = "\n".join(lines) + "\n"
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(text)
    print(f"Appended {dataset_label} ({df.shape[1]} cols) → {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def inspect_one(dataset: str, out_path: Path) -> int:
    """Dispatch per-dataset loader + write_metadata_block call."""
    if dataset not in DATASET_CONFIG:
        sys.stderr.write(f"ERROR: unknown --dataset: {dataset}\n")
        return 2
    loader, proposed, label = DATASET_CONFIG[dataset]
    try:
        df = loader()
    except FileNotFoundError as exc:
        sys.stderr.write(f"ERROR: input missing for {dataset}: {exc}\n")
        return 2
    write_metadata_block(out_path, label, df, proposed)
    return 0


def parse_args() -> argparse.Namespace:
    """Parse CLI args: --dataset (single dataset name or 'all'), --out."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS + ("all",),
        help="Which dataset to inspect (or 'all' for all 11)",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output proposal MD path")
    return p.parse_args()


def main() -> int:
    """Run inspect_one for the requested dataset (or all 11 if --dataset=all)."""
    args = parse_args()
    if args.dataset == "all":
        rc = 0
        for d in DATASETS:
            print(f"\n--- {d} ---", flush=True)
            rc |= inspect_one(d, args.out)
        return rc
    return inspect_one(args.dataset, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
