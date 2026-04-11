"""Download HTAN WUSTL snMultiome data from Synapse.

Downloads Level 3 (counts + fragments) and Level 4 (annotations) files
for 136 snMultiome samples from Terekhanova et al. (Nature 2023).

Prerequisites:
    pip install synapseclient
    synapse login  # or set SYNAPSE_AUTH_TOKEN env var

Usage:
    python scripts/pan_cancer_data/download_htan_multiome.py [--dry-run] [--manifest PATH]
"""

import argparse
import os
import sys

try:
    import synapseclient
except ImportError:
    print("ERROR: synapseclient not installed. Run: pip install synapseclient")
    sys.exit(1)

DEFAULT_MANIFEST = "data/htan_download_manifest.tsv"
DEFAULT_OUTPUT = "/nfs/team205/vk7/sanger_projects/large_data/pan_cancer_multiome"

# Map data_type to subdirectory structure
SUBDIR_MAP = {
    "fragment": "level3/{sample_id}",
    "matrix": "level3/{sample_id}",
    "features": "level3/{sample_id}",
    "barcodes": "level3/{sample_id}",
    "atac_annotation": "level4/atac",
    "rna_annotation": "level4/rna",
}


def parse_manifest(manifest_path):
    """Parse TSV manifest into list of dicts."""
    rows = []
    with open(manifest_path) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            vals = line.strip().split("\t")
            row = dict(zip(header, vals, strict=False))
            rows.append(row)
    return rows


def get_local_path(row, output_dir):
    """Determine local file path for a manifest row."""
    data_type = row["data_type"]
    sample_id = row["sample_id"]
    subdir_template = SUBDIR_MAP.get(data_type, "other")
    subdir = subdir_template.format(sample_id=sample_id)
    return os.path.join(output_dir, subdir, row["filename"])


def download_files(manifest_path, output_dir, dry_run=False):
    """Download all files from manifest via Synapse."""
    rows = parse_manifest(manifest_path)
    print(f"Manifest: {len(rows)} files from {manifest_path}")

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
        if os.path.exists(local_path):
            already_exist += 1
        else:
            to_download.append(row)

    print(f"\nAlready downloaded: {already_exist}")
    print(f"To download: {len(to_download)}")

    if dry_run:
        print("\n[DRY RUN] Would download:")
        for row in to_download[:20]:
            print(f"  {row['synapse_id']} -> {get_local_path(row, output_dir)}")
        if len(to_download) > 20:
            print(f"  ... and {len(to_download) - 20} more")
        return

    if not to_download:
        print("Nothing to download.")
        return

    # Login to Synapse
    print("\nConnecting to Synapse...")
    syn = synapseclient.Synapse()
    syn.login(silent=True)
    print("Authenticated.")

    # Create output directories
    dirs_created = set()
    for row in to_download:
        local_path = get_local_path(row, output_dir)
        d = os.path.dirname(local_path)
        if d not in dirs_created:
            os.makedirs(d, exist_ok=True)
            dirs_created.add(d)

    # Download
    failed = []
    for i, row in enumerate(to_download, 1):
        local_path = get_local_path(row, output_dir)
        local_dir = os.path.dirname(local_path)
        syn_id = row["synapse_id"]
        print(f"[{i}/{len(to_download)}] {syn_id} -> {local_path}")
        try:
            entity = syn.get(syn_id, downloadLocation=local_dir)
            # Synapse may download with original name; rename if needed
            downloaded_path = entity.path
            if downloaded_path != local_path and os.path.exists(downloaded_path):
                os.rename(downloaded_path, local_path)
        except Exception as e:  # noqa: BLE001 — surface all Synapse/IO errors without aborting
            print(f"  FAILED: {e}")
            failed.append((row, str(e)))

    print(f"\nDone. Downloaded: {len(to_download) - len(failed)}, Failed: {len(failed)}")
    if failed:
        print("\nFailed downloads:")
        for row, err in failed:
            print(f"  {row['synapse_id']} ({row['filename']}): {err}")


def main():
    """Download all files from the HTAN manifest via Synapse."""
    parser = argparse.ArgumentParser(description="Download HTAN snMultiome data from Synapse")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Path to download manifest TSV")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded without downloading")
    args = parser.parse_args()

    download_files(args.manifest, args.output_dir, args.dry_run)


if __name__ == "__main__":
    main()
