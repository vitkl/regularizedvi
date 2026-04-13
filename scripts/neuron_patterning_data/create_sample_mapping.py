"""Create sample_mapping.csv for GSE249572 neuron patterning multiome data.

Reads the download manifest and generates a per-sample mapping with
file paths for RNA h5ad and ATAC fragments.

Usage:
    python scripts/neuron_patterning_data/create_sample_mapping.py [--manifest PATH] [--data-dir PATH]
"""

import argparse
import csv
import os
from collections import defaultdict

DEFAULT_MANIFEST = "data/gse249572_download_manifest.tsv"
DEFAULT_DATA_DIR = "/nfs/team283/vk7/sanger_projects/large_data/neuron_patterning_multiome/GSE249572"

SAMPLE_DESCRIPTIONS = {
    "iPSC_prepat_d5_1": "NGN2 prepatterning multiome replicate 1 (day 5)",
    "iPSC_prepat_d5_2": "NGN2 prepatterning multiome replicate 2 (day 5)",
    "iPSC_ctrl": "iPSC control multiome",
}


def main():
    """Parse manifest and write sample_mapping.csv for GSE249572 multiome."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    # Parse manifest
    samples = defaultdict(
        lambda: {
            "sample_id": "",
            "gsm_rna": "",
            "gsm_atac": "",
            "rna_h5ad_path": "",
            "fragment_file_path": "",
            "sample_description": "",
        }
    )

    with open(args.manifest) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sid = row["sample_id"]
            s = samples[sid]
            s["sample_id"] = sid
            s["sample_description"] = SAMPLE_DESCRIPTIONS.get(sid, "")

            if row["data_type"] == "rna_h5ad":
                s["gsm_rna"] = row["gsm_id"]
                # h5ad files are gunzipped after download
                s["rna_h5ad_path"] = os.path.join(args.data_dir, "rna", sid, f"adata_{sid}.h5ad")
            elif row["data_type"] == "atac_fragment":
                s["gsm_atac"] = row["gsm_id"]
                s["fragment_file_path"] = os.path.join(args.data_dir, "atac", sid, "atac_fragments.tsv.gz")

    # Write CSV
    out_path = os.path.join(args.data_dir, "sample_mapping.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = [
        "sample_id",
        "gsm_rna",
        "gsm_atac",
        "rna_h5ad_path",
        "fragment_file_path",
        "sample_description",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sid in sorted(samples):
            writer.writerow(samples[sid])

    print(f"Written {len(samples)} samples to {out_path}")
    for sid in sorted(samples):
        s = samples[sid]
        rna_exists = "OK" if os.path.exists(s["rna_h5ad_path"]) else "MISSING"
        atac_exists = "OK" if os.path.exists(s["fragment_file_path"]) else "MISSING"
        print(f"  {sid}: RNA [{rna_exists}] ATAC [{atac_exists}]")


if __name__ == "__main__":
    main()
