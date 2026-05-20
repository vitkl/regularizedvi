"""Inspect cell-type annotations for 6 labelled v2 datasets; emit diff block proposals.

Phase 0 step 0.5a (+ 0.5b conditional HDMA) of immune_integration_v2 (see
[plan](../../.claude/plans/implement-these-steps-in-tranquil-parasol.md)).

Per dataset, reads only the metadata source (no expression matrix), computes value_counts of
the cell-type column, applies a pre-filled `HARMONIZATION_MAP` to propose a v1
`annotation_hierarchy.md` label (or `—` to drop), and writes a 4-column markdown diff block
to `annotation_harmonization_proposed.md`:

    | Original label | n cells | Proposed harmonized_annotation | Immune keep? |

The user reviews the proposal, edits the harmonized_annotation cells in place, and copies
approved rows into canonical `docs/notebooks/immune_integration_v2/annotation_harmonization.md`
in v1 4-column format (`original_label | harmonized_name | source_dataset | source_column`).

Datasets supported: htan_pan_cancer, gbm_space, hippocampus_aging, lung_smoking,
intestine_hickey, hdma_immune (requires HDMA/annotations/cluster_to_cell_type.csv from
pre-Phase-0 handoff).
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
    "docs/notebooks/immune_integration_v2/annotation_harmonization_proposed.md"
)

DATASETS = (
    "htan_pan_cancer",
    "gbm_space",
    "hippocampus_aging",
    "lung_smoking",
    "intestine_hickey",
    "hdma_immune",
)

# ---------------------------------------------------------------------------
# Per-dataset HARMONIZATION_MAP: Original label → proposed v1 harmonized_name.
# Use the closest existing entry in annotation_hierarchy.md. If a v1 entry does
# not exist, propose a NEW label (e.g. "Macrophage", "Microglia") that the user
# will need to add to annotation_hierarchy.md when approving. Use "—" to drop.
# ---------------------------------------------------------------------------

HTAN_MAP: dict[str, str] = {
    "Macrophages": "Macrophage",  # NEW — add to hierarchy
    "T-cells": "T-cell",  # NEW — generic T-cell (HTAN does not distinguish CD4/CD8)
    "Plasma": "Plasma B cell",
    "B-cells": "B-cell",  # NEW — generic B (HTAN does not distinguish Naive/Memory)
    "DC": "cDC",  # closest v1 match
    "Microglia": "Microglia",  # NEW — add to hierarchy
    "Mast": "Mast cell",  # NEW — add to hierarchy
    # Non-immune (drop):
    "Tumor": "—",
    "Fibroblasts": "—",
    "Endothelial": "—",
    "Hepatocytes": "—",
    "Oligodendrocytes": "—",
    "Islets": "—",
    "Ductal": "—",
}
HTAN_IMMUNE_KEEP = {"Macrophages", "T-cells", "Plasma", "B-cells", "DC", "Microglia", "Mast"}

GBM_MAP: dict[str, str] = {
    # Immune — TAM variants (all NEW, need to be added to annotation_hierarchy.md):
    "Pro-inflammatory TAMs": "TAM Pro-inflammatory",
    "Anti-inflammatory TAMs": "TAM Anti-inflammatory",
    "Resident-TAMs": "TAM Resident",
    "Resident BAM TAMs": "TAM Resident BAM",
    "Angiogenic TAMs": "TAM Angiogenic",
    "RTN1+ TAMs": "TAM RTN1+",
    "Astrocyte-like TAMs": "TAM Astrocyte-like",
    "Interferon TAMs": "TAM Interferon",
    "Proliferative TAMs": "TAM Proliferative",
    "Stress-response TAMs": "TAM Stress-response",
    "Ambiguous (TAMs)": "TAM Ambiguous",
    # Immune — Monocytes / Microglia:
    "Monocytes": "Monocyte",  # NEW generic
    # Immune — T cells:
    "CD8+ T cells (cytotoxic)": "CD8+ T",
    "CD4+ TEM cells": "CD4+ T effector memory",
    "CD8+ T cells": "CD8+ T",
    "HSP-response T cells": "CD4+ T",  # closest in v1
    "Naïve T cells": "CD4+ T naive",
    "T reg": "Treg",
    "IFN-response T cells": "IFN-responding T",
    "Proliferative T cells": "CD8+ T proliferating",
    "Ambiguous (lymphocyte)": "T-cell",  # NEW generic
    # Immune — NK / DC / B / Plasma:
    "NK cells 1": "NK",
    "NK cells 2": "NK",
    "Dendritic cells": "cDC",
    "B cells": "Naive B",
    "Plasma cells": "Plasma B cell",
    # Non-immune (drop) — explicit per observed labels:
    "AC progenitor-like 1": "—",
    "AC progenitor-like 2": "—",
    "AC progenitor-like 3": "—",
    "AC progenitor-like 4": "—",
    "OPC-like 1": "—",
    "OPC-like 2": "—",
    "OPC-like 3": "—",
    "OPC-like 4": "—",
    "OPC-like 5": "—",
    "OPCs": "—",
    "OPCs (differentiating)": "—",
    "OPC-NPC-like 1": "—",
    "OPC-NPC-like 2": "—",
    "OPC-NPC-like 3": "—",
    "OPC-neuronal-like": "—",
    "Oligodendrocytes 1": "—",
    "Oligodendrocytes 2": "—",
    "Ambiguous (oligo.)": "—",
    "Hypoxic 1": "—",
    "Hypoxic 2": "—",
    "NPC-neuronal-like 1": "—",
    "NPC-neuronal-like 2": "—",
    "NPC-neuronal-like 3": "—",
    "NPC-neuronal-like 4": "—",
    "NPC-neuronal-like 5": "—",
    "Proliferative AC-OPC-like": "—",
    "Proliferative NPC-OPC-like": "—",
    "Proliferative nIPC-like": "—",
    "Gliosis-like": "—",
    "AC-gliosis-like 1": "—",
    "AC-gliosis-like 2": "—",
    "AC-gliosis-like 3": "—",
    "AC-gliosis-like 4": "—",
    "Inflammatory astrocytes 1": "—",
    "Inflammatory astrocytes 2": "—",
    "Neural-support astrocytes": "—",
    "Homeostatic astrocytes 2": "—",
    "Deep layer astrocytes": "—",
    "Ambiguous (astrocytes)": "—",
    "Exc L2-3 IT": "—",
    "Exc L4-5 IT": "—",
    "Exc L4 IT": "—",
    "Exc L5 ET": "—",
    "Exc L5-6 NP": "—",
    "Exc L6 IT": "—",
    "Exc L6 Car3": "—",
    "Exc L6b": "—",
    "Exc L6CT": "—",
    "Inh SST": "—",
    "Inh SST (Chodl)": "—",
    "Inh VIP": "—",
    "Inh PVALB": "—",
    "Inh PAX6": "—",
    "Inh LAMP5": "—",
    "Inh Chandelier": "—",
    "Inh RELN": "—",
    "Ambiguous (neurons)": "—",
    "VLMC": "—",
    "Pericytes 1": "—",
    "Pericytes 2": "—",
    "Endothelial (capillary)": "—",
    "Endothelial (arteriole)": "—",
    "Endothelial (venule)": "—",
    "Endothelial (CNA-associated)": "—",
    "Endothelial (Other)": "—",
    "Ambiguous (vascular)": "—",
    "Undefined 1": "—",
    "Undefined 2": "—",
    "Undefined 3": "—",
    "Undefined 4": "—",
    "Undefined 5": "—",
    "Undefined 6": "—",
    "Undefined 7": "—",
}
GBM_IMMUNE_KEEP = {k for k, v in GBM_MAP.items() if v != "—"}

HIPPOCAMPUS_MAP: dict[str, str] = {
    "Microglia": "Microglia",  # NEW
    "Macro": "Macrophage",  # NEW — perivascular macrophages
    "T-Cell": "T-cell",  # NEW generic
    # Non-immune (drop):
    "Oligo": "—",
    "Astro": "—",
    "OPC": "—",
    "SUB": "—",
    "SST": "—",
    "VIP": "—",
    "DG": "—",
    "PVALB": "—",
    "LAMP5": "—",
    "CA1": "—",
    "NR2F2": "—",
    "CA2-CA3": "—",
    "Chandelier": "—",
    "VLMC": "—",
    "Endo": "—",
}
HIPPOCAMPUS_IMMUNE_KEEP = {"Microglia", "Macro", "T-Cell"}

# lung_smoking: Seurat meta.data already includes a per-cell `CellType` column
# (verified by inspecting lung_smoking_meta.csv: T, NK, Macrophage, Monocyte, B, DC,
# NK_T, plus 17 non-immune lung cell types). No need to map via SData4 XLSX.
LUNG_SMOKING_MAP: dict[str, str] = {
    "NK": "NK",
    "T": "T-cell",
    "Macrophage": "Macrophage",
    "Monocyte": "Monocyte",
    "B": "Naive B",  # closest v1; lung_smoking has only one B class
    "DC": "cDC",
    "NK_T": "NK",  # merge with NK
    # Non-immune (drop):
    "AT1": "—",
    "AT2": "—",
    "AT1_AT2": "—",
    "AT2_pro": "—",
    "Club": "—",
    "Ciliated": "—",
    "Goblet": "—",
    "Basal": "—",
    "Lymphatic": "—",
    "Artery": "—",
    "Vein": "—",
    "Capillary": "—",
    "Fibroblast": "—",
    "MyoFib": "—",
    "SMC": "—",
    "Mesothelial": "—",
}
LUNG_SMOKING_IMMUNE_KEEP = {"NK", "T", "Macrophage", "Monocyte", "B", "DC", "NK_T"}

# intestine_hickey: per-cell CellType has 42 fine levels; CellTypeInitial 3 levels (Immune / Epithelial / Stromal).
# We provide a stub mapping for fine CellType — user must extend based on actual value_counts after first run.
HICKEY_MAP: dict[str, str] = {
    # CellTypeInitial coarse:
    "Immune": "<see fine>",
    "Epithelial": "—",
    "Stromal": "—",
    # CellType fine — populate from actual data on first run; placeholder for known immune labels:
    "T cell": "T-cell",
    "B cell": "Naive B",
    "Plasma cell": "Plasma B cell",
    "Macrophage": "Macrophage",
    "Monocyte": "Monocyte",
    "Dendritic cell": "cDC",
    "NK cell": "NK",
    "Mast cell": "Mast cell",
    "ILC": "ILC",
}
HICKEY_IMMUNE_KEEP = {"Immune"}  # CellTypeInitial-level filter

HDMA_MAP: dict[str, str] = {}  # populated at runtime from cluster_to_cell_type.csv
HDMA_IMMUNE_KEEP: set[str] = set()

# ---------------------------------------------------------------------------
# Per-dataset loaders: return pd.Series(value_counts) of the cell-type column.
# ---------------------------------------------------------------------------


def load_htan() -> pd.Series:
    """Load HTAN unified annotations and return value_counts of cell_type."""
    unified = DATA_ROOT / "pan_cancer_multiome/annotations/pan_cancer_multiome_unified_annotations.csv"
    if not unified.is_file():
        raise FileNotFoundError(f"Run build_unified_annotations.py first; missing {unified}")
    df = pd.read_csv(unified, usecols=["cell_type"], low_memory=False)
    return df["cell_type"].fillna("<NaN>").value_counts()


def _h5py_categorical(node) -> list[str]:
    """Decode an h5py categorical node (categories + codes) to a list of strings."""
    cats = [c.decode() if isinstance(c, bytes) else c for c in node["categories"][:]]
    codes = node["codes"][:]
    return [cats[c] if c >= 0 else "<NA>" for c in codes]


def load_gbm() -> pd.Series:
    """Load GBM-Space h5ad obs/annotation_granular via h5py and return value_counts."""
    h5ad = DATA_ROOT / "gbm/GBM_space_snRNA.h5ad"
    if not h5ad.is_file():
        raise FileNotFoundError(f"missing {h5ad}")
    with h5py.File(h5ad, "r") as f:
        if "annotation_granular" not in f["obs"]:
            raise KeyError("obs/annotation_granular not in h5ad")
        vals = _h5py_categorical(f["obs/annotation_granular"])
    return pd.Series(vals, name="annotation_granular").value_counts()


def load_hippocampus() -> pd.Series:
    """Load hippocampus_aging metadata TSV and return value_counts of `subclass`."""
    tsv = (
        DATA_ROOT
        / "hippocampus_aging/annotations/GSE278576_hippocampus_RNA_seurat_object_filtered_cells_metadata.tsv.gz"
    )
    if not tsv.is_file():
        raise FileNotFoundError(f"missing {tsv}")
    df = pd.read_csv(tsv, sep="\t", usecols=["subclass"], low_memory=False)
    return df["subclass"].fillna("<NaN>").value_counts()


def load_lung_smoking() -> pd.Series:
    """Load lung_smoking Seurat meta.data CSV and return value_counts of `CellType`."""
    csv = DATA_ROOT / "lung_smoking/annotations/lung_smoking_meta.csv"
    if not csv.is_file():
        raise FileNotFoundError(f"Run extract_lung_smoking_metadata.R (Slurm) first; missing {csv}")
    df = pd.read_csv(csv, usecols=["CellType"], low_memory=False)
    return df["CellType"].fillna("<NaN>").value_counts()


def load_hickey() -> pd.Series:
    """Concat 8 intestine_hickey compartment metadata CSVs; return DataFrame for caller to branch."""
    folder = DATA_ROOT / "intestine_hickey/annotations"
    csvs = sorted(folder.glob("*_metadata.csv"))
    csvs = [p for p in csvs if "atac_sample_location" not in p.name and "sample_location" not in p.name]
    if not csvs:
        raise FileNotFoundError(f"no *_metadata.csv in {folder}")
    pieces = []
    for p in csvs:
        # Each compartment CSV has barcode + orig.ident + CellType + Multiome cols.
        usecols_try = ["CellType", "CellTypeInitial", "Multiome"]
        try:
            df = pd.read_csv(p, low_memory=False)
            keep = [c for c in usecols_try if c in df.columns]
            df = df[keep].copy()
            df["_source"] = p.stem
            pieces.append(df)
        except (pd.errors.ParserError, OSError, UnicodeDecodeError) as exc:
            print(f"WARN: failed to read {p}: {exc}", file=sys.stderr)
    df = pd.concat(pieces, ignore_index=True)
    # Show BOTH CellTypeInitial (coarse) and CellType (fine) — caller writes both blocks.
    return df  # special-case: returns DataFrame, not Series; caller branches.


def load_hdma() -> pd.Series:
    """Join HDMA per_cell_meta ⨝ cluster_to_cell_type, restrict to SP/TM/LI, return value_counts."""
    per_cell = DATA_ROOT / "HDMA/annotations/per_cell_meta.csv"
    cluster_map = DATA_ROOT / "HDMA/annotations/cluster_to_cell_type.csv"
    if not per_cell.is_file():
        raise FileNotFoundError(f"missing {per_cell}")
    if not cluster_map.is_file():
        raise FileNotFoundError(
            f"missing {cluster_map} — run handoff-hdma-cluster-discovery first OR "
            f"degrade HDMA to label-less (skip hdma_immune)"
        )

    per_cell_df = pd.read_csv(per_cell)
    cm = pd.read_csv(cluster_map)
    # Restrict to immune-rich PCW tissues: Spleen / Thymus / Liver
    per_cell_df = per_cell_df[per_cell_df["organ_code"].isin(["SP", "TM", "LI"])].copy()

    merged = per_cell_df.merge(cm, on="Cluster", how="left", validate="many_to_one")
    if merged["cell_type"].isna().any():
        n_missing = merged["cell_type"].isna().sum()
        missing_clusters = sorted(merged.loc[merged["cell_type"].isna(), "Cluster"].unique())
        print(
            f"WARN: {n_missing:,} HDMA cells in {len(missing_clusters)} clusters lack a cell_type "
            f"mapping: {missing_clusters[:10]}{'...' if len(missing_clusters) > 10 else ''}",
            file=sys.stderr,
        )

    # Populate module-level HDMA_MAP from the loaded CSV
    HDMA_MAP.clear()
    for _, row in cm.iterrows():
        HDMA_MAP[row["cell_type"]] = row["cell_type"]  # 1:1 by default; user edits as needed
        if "immune_keep" in cm.columns and str(row["immune_keep"]).lower() == "yes":
            HDMA_IMMUNE_KEEP.add(row["cell_type"])

    return merged["cell_type"].fillna("<unmapped>").value_counts()


# ---------------------------------------------------------------------------
# Diff-block writer
# ---------------------------------------------------------------------------


def write_diff_block(
    out_path: Path,
    dataset_label: str,
    source_column: str,
    counts: pd.Series,
    harm_map: dict[str, str],
    immune_keep_set: set[str],
    extra_note: str | None = None,
) -> None:
    """Append a 4-column markdown diff block (Original / n cells / Proposed / Immune keep?) to out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = []
    lines.append(f"\n## {dataset_label} — `{source_column}` (proposed by inspect_dataset_annotations.py {timestamp})\n")
    if extra_note:
        lines.append(f"_{extra_note}_\n")
    lines.append("| Original label | n cells | Proposed harmonized_annotation | Immune keep? |")
    lines.append("|---|---:|---|---|")
    for label, n in counts.items():
        prop = harm_map.get(label, "<TODO: not in HARMONIZATION_MAP>")
        keep = "yes" if label in immune_keep_set and prop != "—" else "no"
        # Escape pipes in label
        safe_label = str(label).replace("|", "\\|")
        lines.append(f"| {safe_label} | {n:,} | {prop} | {keep} |")
    text = "\n".join(lines) + "\n"
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(text)
    print(f"Appended {len(counts)} rows to {out_path} (dataset: {dataset_label})", flush=True)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def inspect_one(dataset: str, out_path: Path) -> int:
    """Dispatch per-dataset inspection to the right loader + write_diff_block call."""
    if dataset == "htan_pan_cancer":
        counts = load_htan()
        write_diff_block(
            out_path,
            "HTAN pan-cancer",
            "cell_type (RNA, INNER-joined RNA∩ATAC)",
            counts,
            HTAN_MAP,
            HTAN_IMMUNE_KEEP,
            extra_note=(
                "Source: pan_cancer_multiome_unified_annotations.csv (produced by build_unified_annotations.py)."
            ),
        )
        return 0

    if dataset == "gbm_space":
        counts = load_gbm()
        write_diff_block(
            out_path,
            "GBM-Space",
            "annotation_granular",
            counts,
            GBM_MAP,
            GBM_IMMUNE_KEEP,
            extra_note=(
                "Source: GBM_space_snRNA.h5ad obs/annotation_granular (h5py-only read). "
                "All 155 samples; no 118-subset filter."
            ),
        )
        return 0

    if dataset == "hippocampus_aging":
        counts = load_hippocampus()
        write_diff_block(
            out_path,
            "hippocampus_aging",
            "subclass",
            counts,
            HIPPOCAMPUS_MAP,
            HIPPOCAMPUS_IMMUNE_KEEP,
            extra_note=("Source: GSE278576 RNA seurat_object filtered_cells_metadata.tsv.gz."),
        )
        return 0

    if dataset == "lung_smoking":
        counts = load_lung_smoking()
        write_diff_block(
            out_path,
            "lung_smoking",
            "CellType (from Seurat meta.data)",
            counts,
            LUNG_SMOKING_MAP,
            LUNG_SMOKING_IMMUNE_KEEP,
            extra_note=(
                "Source: GSE241468 Seurat meta.data (extracted by extract_lung_smoking_metadata.R). "
                "The author-provided meta.data already contains a per-cell CellType column — "
                "no SData4 cluster mapping needed."
            ),
        )
        return 0

    if dataset == "intestine_hickey":
        df = load_hickey()
        # Coarse block (CellTypeInitial — 3 levels: Immune / Epithelial / Stromal)
        if "CellTypeInitial" in df.columns:
            counts_coarse = df["CellTypeInitial"].fillna("<NaN>").value_counts()
            write_diff_block(
                out_path,
                "intestine_hickey (coarse)",
                "CellTypeInitial",
                counts_coarse,
                HICKEY_MAP,
                HICKEY_IMMUNE_KEEP,
                extra_note=(
                    "Source: union of 8 *_metadata.csv compartment files. "
                    "Filter applied at loader-time: CellTypeInitial==Immune AND Multiome==Yes."
                ),
            )
        # Fine block (CellType — 42 levels)
        if "CellType" in df.columns:
            # Only show fine labels for cells where CellTypeInitial == Immune
            if "CellTypeInitial" in df.columns:
                df_im = df[df["CellTypeInitial"] == "Immune"]
            else:
                df_im = df
            counts_fine = df_im["CellType"].fillna("<NaN>").value_counts()
            write_diff_block(
                out_path,
                "intestine_hickey (fine immune)",
                "CellType",
                counts_fine,
                HICKEY_MAP,
                set(HICKEY_MAP.keys()),
                extra_note="Fine CellType (subset to CellTypeInitial==Immune).",
            )
        return 0

    if dataset == "hdma_immune":
        counts = load_hdma()
        # Re-derive immune-keep from the cluster_to_cell_type.csv via HDMA_IMMUNE_KEEP populated in load_hdma.
        write_diff_block(
            out_path,
            "HDMA Spleen/Thymus/Liver",
            "Cluster → cell_type (via cluster_to_cell_type.csv)",
            counts,
            HDMA_MAP,
            HDMA_IMMUNE_KEEP,
            extra_note=(
                "Source: per_cell_meta.csv ⨝ cluster_to_cell_type.csv (from pre-Phase-0 handoff). "
                "Restricted to organ_code ∈ {SP, TM, LI}."
            ),
        )
        return 0

    sys.stderr.write(f"ERROR: unknown --dataset: {dataset}\n")
    return 2


def parse_args() -> argparse.Namespace:
    """Parse CLI args: --dataset, --out."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True, choices=DATASETS, help="Which dataset to inspect")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output proposal MD path")
    return p.parse_args()


def main() -> int:
    """Run inspect_one for the requested dataset."""
    args = parse_args()
    return inspect_one(args.dataset, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
