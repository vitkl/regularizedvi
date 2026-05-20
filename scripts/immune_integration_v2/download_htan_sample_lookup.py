"""Download HTAN Ding-lab Sample_ID_Lookup table → CSV.

Phase 0 step 0.2 of immune_integration_v2 (see [plan](../../.claude/plans/implement-these-steps-in-tranquil-parasol.md)).

Source: Ding-lab `PanCan_snATAC_publication` GitHub repo. The XLSX maps `Piece_ID` (used in
HTAN per-cell annotation CSVs) to donor / case / cancer-type metadata.

URL is inline-pinned as `DING_LAB_URL` per grill resolution (verified via gh api on 2026-05-20,
SHA 685299ef93abac05db1f6288a439dd6d6a1a11a7).

XLSX layout (both 'ATAC data' and 'RNA data' sheets share the same structure):
- Rows 0-4: link references (HTAN DCC, CPTAC GEO/GDC/dbGaP, etc.) — skip
- Row 5: blank — skip
- Row 6: column headers
- Rows 7+: actual data (one row per Piece_ID)

We load BOTH sheets, dedup on `Piece_ID_ATAC` (same piece IDs appear in both for multiome
samples), and emit a single CSV with the columns useful for downstream loaders.

NB: the XLSX does NOT contain sex / age / tumor_stage — those columns referenced in the
parent plan are not in this Ding-lab artefact. Only `piece_id, cancer_type, participant_id
(donor), biospecimen_id (case), geo_sample_name` are available here.

Output: `pan_cancer_multiome_sample_lookup.csv`.
"""

from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path

import pandas as pd
import requests

DING_LAB_URL = (
    "https://raw.githubusercontent.com/ding-lab/PanCan_snATAC_publication/main/"
    "Sample_ID_Lookup_table_in_repositories.xlsx"
)
DEFAULT_OUT = (
    "/nemo/lab/briscoej/home/users/kleshcv/large_data/pan_cancer_multiome/"
    "annotations/pan_cancer_multiome_sample_lookup.csv"
)
TIMEOUT_S = 60


def parse_args() -> argparse.Namespace:
    """Parse CLI args: --out, --url."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=DEFAULT_OUT, help="Output CSV path")
    p.add_argument("--url", default=DING_LAB_URL, help="Override download URL (default: inline-pinned)")
    return p.parse_args()


def main() -> int:
    """Fetch Ding-lab XLSX, parse both sheets (skip 7 link rows), dedup, write CSV."""
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"GET {args.url}", flush=True)
    r = requests.get(args.url, timeout=TIMEOUT_S, allow_redirects=True)
    r.raise_for_status()
    body = r.content
    print(f"  bytes={len(body):,}  sha256={hashlib.sha256(body).hexdigest()}", flush=True)

    xls = pd.ExcelFile(io.BytesIO(body))
    print(f"  sheets: {xls.sheet_names}", flush=True)

    # XLSX has 7 link/blank rows before the actual table (verified 2026-05-20).
    # Row 6 = column headers; rows 7+ = per-piece data.
    sheets = []
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name, header=6)
        df["_sheet"] = sheet_name
        # Drop fully-empty rows (some sheets have trailing blanks)
        df = df.dropna(how="all", subset=[c for c in df.columns if c != "_sheet"])
        print(f"  sheet {sheet_name!r}: {len(df):,} data rows, cols={list(df.columns)}", flush=True)
        sheets.append(df)

    combined = pd.concat(sheets, ignore_index=True)
    print(f"  combined: {len(combined):,} rows", flush=True)

    # Dedup on Piece_ID_ATAC (multiome samples appear in both sheets with same Piece_ID).
    # Prefer ATAC sheet (first occurrence) since it has all multiome samples.
    if "Piece_ID_ATAC" in combined.columns:
        before = len(combined)
        combined = combined.drop_duplicates(subset=["Piece_ID_ATAC"], keep="first")
        print(
            f"  after dedup on Piece_ID_ATAC: {len(combined):,} ({before - len(combined)} duplicates dropped)",
            flush=True,
        )
    else:
        print("  WARN: Piece_ID_ATAC column not found; no dedup applied", flush=True)

    # Rename to lowercase snake_case for downstream join.
    rename_map = {
        "Piece_ID_ATAC": "piece_id",
        "Cancer type": "cancer_type",
        "HTAN DCC Participant ID": "donor_id",
        "HTAN DCC Biospecimen ID": "biospecimen_id",
        "GEO sample name": "geo_sample_name",
        "ATAC data type": "atac_data_type",
        "Raw data uploaded to": "raw_data_uploaded_to",
        "Processed data uploaded to": "processed_data_uploaded_to",
        "CDS sample name": "cds_sample_name",
        "GDC bam file ID": "gdc_bam_file_id",
        "_sheet": "source_sheet",
    }
    combined = combined.rename(columns={k: v for k, v in rename_map.items() if k in combined.columns})

    combined.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  ({out_path.stat().st_size / 1e3:.1f} KB)", flush=True)
    print(f"\nColumns: {list(combined.columns)}", flush=True)
    print("\nCancer-type value_counts:", flush=True)
    if "cancer_type" in combined.columns:
        print(combined["cancer_type"].value_counts().to_string(), flush=True)
    print("\nFirst 3 rows:", flush=True)
    print(combined.head(3).to_string(), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
