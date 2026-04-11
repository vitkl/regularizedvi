"""Create canonical 10x symlinks for HTAN multiome count matrices.

scanpy.read_10x_mtx() expects files named exactly `matrix.mtx.gz`, `features.tsv.gz`,
and `barcodes.tsv.gz` in a directory. HTAN files are prefixed with the sample ID
(e.g. `CE336E1-S1-matrix.mtx.gz`). This script adds RELATIVE symlinks with the
canonical names alongside the prefixed originals, leaving provenance intact.

Idempotent: skips sample dirs that already have the canonical symlinks.

Usage:
    python scripts/pan_cancer_data/create_10x_symlinks.py [--data-dir PATH]
"""

import argparse
import os
from pathlib import Path

DEFAULT_DATA_DIR = "/nfs/team205/vk7/sanger_projects/large_data/pan_cancer_multiome"

CANONICAL = ["matrix.mtx.gz", "features.tsv.gz", "barcodes.tsv.gz"]


def main():
    """Add canonical 10x symlinks (matrix.mtx.gz etc.) next to prefixed HTAN files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    level3_dir = Path(args.data_dir) / "level3"
    if not level3_dir.is_dir():
        raise SystemExit(f"Not found: {level3_dir}")

    sample_dirs = sorted(d for d in level3_dir.iterdir() if d.is_dir())
    print(f"Found {len(sample_dirs)} sample directories in {level3_dir}")

    linked = 0
    skipped = 0
    missing = 0

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        for canonical in CANONICAL:
            target_link = sample_dir / canonical
            prefixed = sample_dir / f"{sample_id}-{canonical}"

            if target_link.is_symlink() or target_link.exists():
                skipped += 1
                continue
            if not prefixed.exists():
                print(f"  MISSING: {prefixed}")
                missing += 1
                continue

            # Relative symlink: just the prefixed filename (same dir)
            os.symlink(prefixed.name, target_link)
            linked += 1

    print(f"\nCreated {linked} symlinks, skipped {skipped}, missing {missing}")
    print(f"Total expected: {len(sample_dirs) * 3}")


if __name__ == "__main__":
    main()
