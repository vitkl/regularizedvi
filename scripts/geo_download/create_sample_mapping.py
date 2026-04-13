"""Create sample_mapping.csv for downloaded multiome datasets.

Reads a download manifest and generates a per-sample mapping CSV with
local file paths. Column format matches immune_integration conventions:
  - sample_id, fragment_file_path (always present)
  - RNA columns depend on format: rna_h5ad_path / gex_h5_path /
    (barcodes_path, features_path, matrix_path)

Note: Output is CSV (comma-separated), matching the immune_integration
sample_mapping convention. Download manifests are TSV.

Usage:
    python scripts/geo_download/create_sample_mapping.py \
        --manifest data/gse295277_gse295308_download_manifest.tsv \
        --data-dir /nfs/team205/vk7/sanger_projects/large_data/bcg_trained_immunity
"""

import argparse
import csv
import os
from collections import defaultdict

# Canonical local paths (must match download_multiome.py)
RNA_H5AD_PATH = "rna/{sample_id}/adata.h5ad"  # after gunzip
RNA_H5_PATH = "rna/{sample_id}/filtered_feature_bc_matrix.h5"
RNA_MTX_DIR = "rna/{sample_id}"
ATAC_FRAG_PATH = "atac/{sample_id}/atac_fragments.tsv.gz"
ATAC_INDEX_PATH = "atac/{sample_id}/atac_fragments.tsv.gz.tbi"


def parse_manifest(manifest_path):
    """Parse TSV manifest into list of dicts."""
    rows = []
    with open(manifest_path) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            vals = line.strip().split("\t")
            if len(vals) == len(header):
                rows.append(dict(zip(header, vals, strict=False)))
    return rows


def detect_rna_format(sample_types):
    """Detect RNA data format for a sample from its data types."""
    if "rna_h5ad" in sample_types:
        return "h5ad"
    if "rna_h5" in sample_types:
        return "h5"
    if "rna_mtx" in sample_types:
        return "mtx"
    return None


def main():
    """Parse manifest and write sample_mapping.csv."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    rows = parse_manifest(args.manifest)

    # Group by sample_id
    samples = defaultdict(
        lambda: {
            "data_types": set(),
            "gsm_gex": "",
            "gsm_atac": "",
            "gse_id": "",
            "mtx_files": [],
        }
    )

    for row in rows:
        sid = row["sample_id"]
        s = samples[sid]
        dt = row["data_type"]
        s["data_types"].add(dt)
        s["gse_id"] = row.get("gse_id", "")

        if dt in ("rna_h5ad", "rna_h5", "rna_mtx"):
            s["gsm_gex"] = row["gsm_id"]
            if dt == "rna_mtx":
                s["mtx_files"].append(row["filename"])
        elif dt in ("atac_fragment", "atac_fragment_index"):
            s["gsm_atac"] = row["gsm_id"]

    # Build output rows and collect modality warnings (D1 fix)
    out_rows = []
    warnings = []
    for sid in sorted(samples):
        s = samples[sid]
        rna_fmt = detect_rna_format(s["data_types"])
        has_atac = "atac_fragment" in s["data_types"]

        # D1 fix: warn when sample has only one modality
        if has_atac and rna_fmt is None:
            warnings.append(f"  WARNING: {sid} has ATAC fragments but no RNA data")
        elif not has_atac and rna_fmt is not None:
            warnings.append(f"  WARNING: {sid} has RNA data but no ATAC fragments")

        out = {
            "sample_id": sid,
            "gse_id": s["gse_id"],
            "gsm_gex": s["gsm_gex"],
            "gsm_atac": s["gsm_atac"],
        }

        # Fragment file path (always present, empty string if no ATAC)
        if has_atac:
            out["fragment_file_path"] = os.path.join(args.data_dir, ATAC_FRAG_PATH.format(sample_id=sid))
        else:
            out["fragment_file_path"] = ""

        # Fragment index (always present, empty string if no index)
        if "atac_fragment_index" in s["data_types"]:
            out["fragment_index_path"] = os.path.join(args.data_dir, ATAC_INDEX_PATH.format(sample_id=sid))
        else:
            out["fragment_index_path"] = ""

        # RNA path columns — format-dependent (matching immune_integration conventions)
        # P1 fix: always set RNA columns (empty string when format is None)
        if rna_fmt == "h5ad":
            out["rna_h5ad_path"] = os.path.join(args.data_dir, RNA_H5AD_PATH.format(sample_id=sid))
        elif rna_fmt == "h5":
            out["gex_h5_path"] = os.path.join(args.data_dir, RNA_H5_PATH.format(sample_id=sid))
        elif rna_fmt == "mtx":
            mtx_dir = os.path.join(args.data_dir, RNA_MTX_DIR.format(sample_id=sid))
            barcodes = [f for f in s["mtx_files"] if "barcodes" in f.lower()]
            features = [f for f in s["mtx_files"] if "features" in f.lower()]
            matrix = [f for f in s["mtx_files"] if "matrix" in f.lower()]
            out["barcodes_path"] = os.path.join(mtx_dir, barcodes[0]) if barcodes else ""
            out["features_path"] = os.path.join(mtx_dir, features[0]) if features else ""
            out["matrix_path"] = os.path.join(mtx_dir, matrix[0]) if matrix else ""

        out_rows.append(out)

    # Determine fieldnames from all rows (union of keys, fixed order)
    all_keys = []
    seen = set()
    fixed_order = [
        "sample_id",
        "gse_id",
        "gsm_gex",
        "gsm_atac",
        "fragment_file_path",
        "fragment_index_path",
        "rna_h5ad_path",
        "gex_h5_path",
        "barcodes_path",
        "features_path",
        "matrix_path",
    ]
    for k in fixed_order:
        for row in out_rows:
            if k in row and k not in seen:
                all_keys.append(k)
                seen.add(k)
                break

    # D3 fix: ensure all rows have all expected keys (fill missing with "")
    for row in out_rows:
        for k in all_keys:
            if k not in row:
                row[k] = ""

    # Write CSV (comma-separated — matches immune_integration convention)
    out_path = os.path.join(args.data_dir, "sample_mapping.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        # D3 fix: use extrasaction="raise" to catch schema drift
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="raise")
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    # Report
    print(f"Written {len(out_rows)} samples to {out_path}")
    print(f"Columns: {', '.join(all_keys)}")

    # D1 fix: print modality warnings
    if warnings:
        print(f"\nModality warnings ({len(warnings)}):")
        for w in warnings:
            print(w)

    print()
    for row in out_rows:
        frag_ok = "OK" if row.get("fragment_file_path") and os.path.exists(row["fragment_file_path"]) else "MISSING"
        rna_path = row.get("rna_h5ad_path") or row.get("gex_h5_path") or row.get("matrix_path") or ""
        rna_ok = "OK" if rna_path and os.path.exists(rna_path) else "MISSING"
        print(f"  {row['sample_id']}: RNA [{rna_ok}] ATAC [{frag_ok}]")


if __name__ == "__main__":
    main()
