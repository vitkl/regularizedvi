"""Build HTAN pan-cancer unified annotations via RNA∩ATAC INNER join.

Phase 0 step 0.1 of immune_integration_v2 (see [plan](../../.claude/plans/implement-these-steps-in-tranquil-parasol.md)).

Join key: the `barcode` column in both CSVs encodes `{cancer_type}_{piece_id_stripped}_{original_barcode}`.
INNER join on `barcode` yields the set of cells with PAIRED RNA + ATAC measurements (multiome intersection).

Output: `pan_cancer_multiome_unified_annotations.csv` with RNA columns verbatim + ATAC columns suffixed `_atac`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

DEFAULT_RNA = "/nemo/lab/briscoej/home/users/kleshcv/large_data/pan_cancer_multiome/annotations/pan_cancer_multiome_rna_annotations.csv"
DEFAULT_ATAC = "/nemo/lab/briscoej/home/users/kleshcv/large_data/pan_cancer_multiome/annotations/pan_cancer_multiome_atac_annotations.csv"
DEFAULT_OUT = "/nemo/lab/briscoej/home/users/kleshcv/large_data/pan_cancer_multiome/annotations/pan_cancer_multiome_unified_annotations.csv"


def parse_args() -> argparse.Namespace:
    """Parse CLI args: --rna, --atac, --out."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rna", default=DEFAULT_RNA, help="HTAN RNA annotations CSV")
    p.add_argument("--atac", default=DEFAULT_ATAC, help="HTAN ATAC annotations CSV")
    p.add_argument("--out", default=DEFAULT_OUT, help="Output unified CSV")
    return p.parse_args()


def main() -> int:
    """Build HTAN unified annotations via RNA∩ATAC inner-join on `barcode`."""
    args = parse_args()
    rna_path, atac_path, out_path = Path(args.rna), Path(args.atac), Path(args.out)

    for p in (rna_path, atac_path):
        if not p.is_file():
            sys.stderr.write(f"ERROR: input not found: {p}\n")
            return 2

    print(f"Reading RNA  : {rna_path}", flush=True)
    rna = pd.read_csv(rna_path, low_memory=False)
    print(f"  rows={len(rna):,}  cols={len(rna.columns)}  unique Piece_ID={rna['Piece_ID'].nunique()}", flush=True)

    print(f"Reading ATAC : {atac_path}", flush=True)
    atac = pd.read_csv(atac_path, low_memory=False)
    print(f"  rows={len(atac):,}  cols={len(atac.columns)}  unique Piece_ID={atac['Piece_ID'].nunique()}", flush=True)

    if "barcode" not in rna.columns or "barcode" not in atac.columns:
        sys.stderr.write("ERROR: both CSVs must have a `barcode` column\n")
        return 2

    atac_suffixed = atac.rename(columns={c: f"{c}_atac" for c in atac.columns if c != "barcode"})

    print("Inner-join on `barcode`...", flush=True)
    merged = rna.merge(atac_suffixed, on="barcode", how="inner", validate="one_to_one")
    print(f"  merged rows={len(merged):,}  cols={len(merged.columns)}", flush=True)

    rna_only = len(rna) - len(merged)
    atac_only = len(atac) - len(merged)
    print(f"  RNA-only (dropped):  {rna_only:,}", flush=True)
    print(f"  ATAC-only (dropped): {atac_only:,}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    if "cell_type" in merged.columns:
        print("\ncell_type value_counts (RNA, top 15):", flush=True)
        print(merged["cell_type"].value_counts().head(15).to_string(), flush=True)
    if "cancer_type_atac" in merged.columns:
        print("\ncancer_type_atac value_counts:", flush=True)
        print(merged["cancer_type_atac"].value_counts().to_string(), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
