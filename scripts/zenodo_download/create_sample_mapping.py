"""Build hdma_sample_mapping.csv from a populated HDMA download tree.

Scans <output_dir>/rna and <output_dir>/atac for sample directories
matching `T<donor>_b<batch>_<organ>_PCW<age>` and writes a CSV with
columns matching the immune_integration sample_mapping convention,
plus HDMA-specific donor/batch/PCW fields parsed from the sample id.

Output columns:
    sample_id, organ, donor_id, batch, PCW,
    barcodes_path, features_path, matrix_path,
    fragment_file_path, fragment_index_path

Usage:
    python scripts/zenodo_download/create_sample_mapping.py \
        --output-dir /nemo/lab/briscoej/home/users/kleshcv/large_data/HDMA
"""

import argparse
import csv
import os
import re
import sys

SAMPLE_RE = re.compile(r"^T(?P<donor>\w+?)_b(?P<batch>\d+)_(?P<organ>[A-Za-z]+)_PCW(?P<pcw>\w+)$")

RNA_FILES = ["barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz"]
ATAC_FRAG = "fragments.tsv.gz"
ATAC_INDEX = "fragments.tsv.gz.tbi"


def discover_samples(output_dir):
    """Return dict sample_id -> {paths...} from rna/ and atac/ trees."""
    rna_dir = os.path.join(output_dir, "rna")
    atac_dir = os.path.join(output_dir, "atac")
    samples = {}

    def _nonempty(path):
        return os.path.exists(path) and os.path.getsize(path) > 0

    if os.path.isdir(rna_dir):
        for sid in sorted(os.listdir(rna_dir)):
            if not SAMPLE_RE.match(sid):
                continue
            samples.setdefault(sid, {})
            for fname in RNA_FILES:
                fpath = os.path.join(rna_dir, sid, fname)
                if _nonempty(fpath):
                    samples[sid][f"rna_{fname.split('.')[0]}_path"] = fpath

    if os.path.isdir(atac_dir):
        for sid in sorted(os.listdir(atac_dir)):
            if not SAMPLE_RE.match(sid):
                continue
            samples.setdefault(sid, {})
            frag = os.path.join(atac_dir, sid, ATAC_FRAG)
            idx = os.path.join(atac_dir, sid, ATAC_INDEX)
            if _nonempty(frag):
                samples[sid]["fragment_file_path"] = frag
            if _nonempty(idx):
                samples[sid]["fragment_index_path"] = idx

    return samples


def main():
    """CLI entry point: scan the HDMA download tree and emit hdma_sample_mapping.csv."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, help="HDMA root with rna/ atac/")
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path. Default: <output-dir>/manifest/hdma_sample_mapping.csv",
    )
    args = parser.parse_args()

    samples = discover_samples(args.output_dir)
    out_path = args.out or os.path.join(args.output_dir, "manifest", "hdma_sample_mapping.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fieldnames = [
        "sample_id",
        "organ",
        "donor_id",
        "batch",
        "PCW",
        "barcodes_path",
        "features_path",
        "matrix_path",
        "fragment_file_path",
        "fragment_index_path",
    ]

    n_complete = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sid in sorted(samples):
            m = SAMPLE_RE.match(sid)
            paths = samples[sid]
            row = {
                "sample_id": sid,
                "organ": m.group("organ"),
                "donor_id": m.group("donor"),
                "batch": m.group("batch"),
                "PCW": m.group("pcw"),
                "barcodes_path": paths.get("rna_barcodes_path", ""),
                "features_path": paths.get("rna_features_path", ""),
                "matrix_path": paths.get("rna_matrix_path", ""),
                "fragment_file_path": paths.get("fragment_file_path", ""),
                "fragment_index_path": paths.get("fragment_index_path", ""),
            }
            writer.writerow(row)
            if all([row["barcodes_path"], row["matrix_path"], row["fragment_file_path"]]):
                n_complete += 1

    print(
        f"Wrote {len(samples)} samples to {out_path}  ({n_complete} complete: have barcodes + matrix + fragments)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
