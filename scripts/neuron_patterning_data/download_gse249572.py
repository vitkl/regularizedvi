"""Download GSE249572 multiome data from GEO FTP.

Downloads 3 multiome sample pairs (RNA h5ad + ATAC fragments) from
Fleck et al. (Science 2024) — iPSC-derived neuron patterning dataset.

Usage:
    python scripts/neuron_patterning_data/download_gse249572.py [--dry-run] [--manifest PATH]
"""

import argparse
import gzip
import os
import shutil
import subprocess

DEFAULT_MANIFEST = "data/gse249572_download_manifest.tsv"
DEFAULT_OUTPUT = "/nfs/team283/vk7/sanger_projects/large_data/neuron_patterning_multiome/GSE249572"

# Map data_type to subdirectory
SUBDIR_MAP = {
    "rna_h5ad": "rna/{sample_id}",
    "atac_fragment": "atac/{sample_id}",
}

# Map data_type to canonical local filename (strip GSM prefix)
CANONICAL_NAME = {
    "rna_h5ad": "adata_{sample_id}.h5ad.gz",
    "atac_fragment": "atac_fragments.tsv.gz",
}


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


def get_local_path(row, output_dir):
    """Determine local file path for a manifest row. Returns None for unknown data_type."""
    data_type = row["data_type"]
    sample_id = row["sample_id"]
    subdir_template = SUBDIR_MAP.get(data_type)
    canonical_template = CANONICAL_NAME.get(data_type)
    if subdir_template is None or canonical_template is None:
        print(f"  WARNING: unknown data_type '{data_type}' for {row.get('gsm_id', '?')}, skipping")
        return None
    subdir = subdir_template.format(sample_id=sample_id)
    canonical = canonical_template.format(sample_id=sample_id)
    return os.path.join(output_dir, subdir, canonical)


def gunzip_file(gz_path, keep_gz=False):
    """Decompress a .gz file, returning the output path."""
    out_path = gz_path[:-3]  # strip .gz
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    if not keep_gz:
        os.remove(gz_path)
    return out_path


def download_files(manifest_path, output_dir, dry_run=False):
    """Download all files from manifest via wget."""
    rows = parse_manifest(manifest_path)
    print(f"Manifest: {len(rows)} files from {manifest_path}")

    # Count by type
    type_counts = {}
    for r in rows:
        type_counts[r["data_type"]] = type_counts.get(r["data_type"], 0) + 1
    for dtype, count in sorted(type_counts.items()):
        print(f"  {dtype}: {count}")

    # Check which files already exist (check both .gz and decompressed for h5ad)
    to_download = []
    already_exist = 0
    for row in rows:
        local_path = get_local_path(row, output_dir)
        if local_path is None:
            continue
        # For h5ad, also check decompressed version
        decompressed = local_path[:-3] if local_path.endswith(".gz") and row["data_type"] == "rna_h5ad" else None
        if os.path.exists(local_path) or (decompressed and os.path.exists(decompressed)):
            already_exist += 1
        else:
            to_download.append(row)

    print(f"\nAlready downloaded: {already_exist}")
    print(f"To download: {len(to_download)}")

    if dry_run:
        print("\n[DRY RUN] Would download:")
        for row in to_download:
            local = get_local_path(row, output_dir)
            print(f"  {row['url']}")
            print(f"    -> {local}")
            if row["data_type"] == "rna_h5ad":
                print(f"    -> gunzip to {local[:-3]}")
        if not to_download:
            print("  (nothing to download)")
        return

    if not to_download:
        print("Nothing to download.")
        return

    # Create output directories
    dirs_created = set()
    for row in to_download:
        local_path = get_local_path(row, output_dir)
        d = os.path.dirname(local_path)
        if d not in dirs_created:
            os.makedirs(d, exist_ok=True)
            dirs_created.add(d)

    # Download with wget -c (resumable)
    failed = []
    for i, row in enumerate(to_download, 1):
        local_path = get_local_path(row, output_dir)
        url = row["url"]
        print(f"\n[{i}/{len(to_download)}] {row['gsm_id']} ({row['data_type']})")
        print(f"  URL:  {url}")
        print(f"  Dest: {local_path}")

        try:
            subprocess.run(
                ["wget", "-c", "-q", "--show-progress", "-O", local_path, url],
                check=True,
            )
            # Gunzip h5ad files
            if row["data_type"] == "rna_h5ad" and local_path.endswith(".gz"):
                print(f"  Decompressing {os.path.basename(local_path)}...")
                out = gunzip_file(local_path)
                print(f"  -> {out}")
        except subprocess.CalledProcessError as e:
            print(f"  FAILED: wget returned {e.returncode}")
            failed.append((row, str(e)))
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {e}")
            failed.append((row, str(e)))

    print(f"\nDone. Downloaded: {len(to_download) - len(failed)}, Failed: {len(failed)}")
    if failed:
        print("\nFailed downloads:")
        for row, err in failed:
            print(f"  {row['gsm_id']} ({row['filename']}): {err}")


def main():
    """Download GSE249572 multiome files from GEO FTP."""
    parser = argparse.ArgumentParser(description="Download GSE249572 multiome data from GEO")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Path to download manifest TSV")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded")
    args = parser.parse_args()

    download_files(args.manifest, args.output_dir, args.dry_run)


if __name__ == "__main__":
    main()
