"""Download multiome data from GEO FTP using a manifest TSV.

Handles ATAC fragment files, RNA h5ad/h5/mtx formats. Uses wget -c
for resumable downloads. Auto-gunzips h5ad files after download.

Usage:
    python scripts/geo_download/download_multiome.py \
        --manifest data/gse295277_gse295308_download_manifest.tsv \
        --output-dir /nfs/team205/vk7/sanger_projects/large_data/bcg_trained_immunity

    python scripts/geo_download/download_multiome.py --manifest ... --dry-run
"""

import argparse
import gzip
import os
import shutil
import subprocess

# Map data_type to subdirectory template
SUBDIR_MAP = {
    "rna_h5ad": "rna/{sample_id}",
    "rna_h5": "rna/{sample_id}",
    "rna_mtx": "rna/{sample_id}",
    "atac_fragment": "atac/{sample_id}",
    "atac_fragment_index": "atac/{sample_id}",
}

# Map data_type to canonical local filename
# {original} keeps the original filename from GEO
CANONICAL_NAME = {
    "rna_h5ad": "adata.h5ad.gz",
    "rna_h5": "filtered_feature_bc_matrix.h5",
    "rna_mtx": "{original}",
    "atac_fragment": "atac_fragments.tsv.gz",
    "atac_fragment_index": "atac_fragments.tsv.gz.tbi",
}

# Data types that should be gunzipped after download
GUNZIP_TYPES = {"rna_h5ad"}


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
    if canonical_template == "{original}":
        canonical = row["filename"]
    else:
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
    print(f"Output:   {output_dir}")

    # Count by type
    type_counts = {}
    for r in rows:
        type_counts[r["data_type"]] = type_counts.get(r["data_type"], 0) + 1
    for dtype, count in sorted(type_counts.items()):
        print(f"  {dtype}: {count}")

    # Check which files already exist
    to_download = []
    already_exist = 0
    for row in rows:
        local_path = get_local_path(row, output_dir)
        if local_path is None:
            continue
        # For gunzip types, also check decompressed version
        decompressed = None
        if row["data_type"] in GUNZIP_TYPES and local_path.endswith(".gz"):
            decompressed = local_path[:-3]
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
            if local is None:
                continue
            print(f"  {row['url']}")
            print(f"    -> {local}")
            if row["data_type"] in GUNZIP_TYPES:
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
        if local_path is None:
            continue
        d = os.path.dirname(local_path)
        if d not in dirs_created:
            os.makedirs(d, exist_ok=True)
            dirs_created.add(d)

    # Download with wget -c (resumable)
    failed = []
    for i, row in enumerate(to_download, 1):
        local_path = get_local_path(row, output_dir)
        if local_path is None:
            continue
        url = row["url"]
        print(f"\n[{i}/{len(to_download)}] {row['gsm_id']} ({row['data_type']})")
        print(f"  URL:  {url}")
        print(f"  Dest: {local_path}")

        try:
            subprocess.run(
                ["wget", "-c", "-q", "--show-progress", "-O", local_path, url],
                check=True,
            )
            # Gunzip applicable files
            if row["data_type"] in GUNZIP_TYPES and local_path.endswith(".gz"):
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
    """Download multiome files from GEO FTP."""
    parser = argparse.ArgumentParser(description="Download multiome data from GEO FTP")
    parser.add_argument("--manifest", required=True, help="Path to download manifest TSV")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded")
    args = parser.parse_args()

    download_files(args.manifest, args.output_dir, args.dry_run)


if __name__ == "__main__":
    main()
