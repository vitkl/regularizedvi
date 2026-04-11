"""Create sample_mapping.csv for pan-cancer multiome data.

Reads the download manifest and generates a per-sample mapping with
file paths for fragments, count matrices, and annotations.

Usage:
    python scripts/pan_cancer_data/create_sample_mapping.py [--manifest PATH] [--data-dir PATH]
"""

import argparse
import csv
import os
from collections import defaultdict

DEFAULT_MANIFEST = "data/htan_download_manifest.tsv"
DEFAULT_DATA_DIR = "/nfs/team205/vk7/sanger_projects/large_data/pan_cancer_multiome"

# Combined annotation CSVs produced by convert_rds_annotations.R
ATAC_COMBINED_CSV = "annotations/pan_cancer_multiome_atac_annotations.csv"
RNA_COMBINED_CSV = "annotations/pan_cancer_multiome_rna_annotations.csv"


def main():
    """Parse the HTAN manifest and write sample_mapping.csv for pan-cancer multiome."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    # Parse manifest — annotations stored as lists to preserve duplicates
    samples = defaultdict(
        lambda: {
            "sample_id": "",
            "organ": "",
            "diagnosis": "",
            "cancer_type": "",
            "fragment_file_path": "",
            "matrix_dir": "",
            "atac_annotation_rds": [],
            "rna_annotation_rds": [],
            "annotation_source": "none",
        }
    )

    with open(args.manifest) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sid = row["sample_id"]
            s = samples[sid]
            s["sample_id"] = sid
            s["organ"] = row["organ"]
            s["diagnosis"] = row["diagnosis"]

            dtype = row["data_type"]
            if dtype == "fragment":
                s["fragment_file_path"] = os.path.join(args.data_dir, "level3", sid, row["filename"])
            elif dtype == "matrix":
                s["matrix_dir"] = os.path.join(args.data_dir, "level3", sid)
            elif dtype == "atac_annotation":
                rds_path = os.path.join(args.data_dir, "level4", "atac", row["filename"])
                s["atac_annotation_rds"].append(rds_path)
            elif dtype == "rna_annotation":
                rds_path = os.path.join(args.data_dir, "level4", "rna", row["filename"])
                s["rna_annotation_rds"].append(rds_path)

    # Infer cancer_type from ATAC annotation filename pattern: {CancerType}_{SampleID}.rds
    # Fall back to sample_id prefix map if no ATAC annotation or no underscore.
    prefix_map = {
        "CE": "CESC",
        "CM": "CRC",
        "CP": "UCEC",
        "GB": "GBM",
        "HT": "BRCA",
        "ML": "SKCM",
        "PM": "PDAC",
        "VF": "OV",
        "SN": "HNSCC",
        "SP": "PDAC",
        "HN": "HNSCC",
    }
    for sid, s in samples.items():
        cancer_type = ""
        # Prefer the cancer-type-prefixed ATAC rds filename
        for rds_path in s["atac_annotation_rds"]:
            rds = os.path.basename(rds_path)
            if "_" in rds:
                cancer_type = rds.split("_")[0]
                break
        if not cancer_type:
            cancer_type = prefix_map.get(sid[:2], "Unknown")
        s["cancer_type"] = cancer_type

    # Determine annotation_source (prefer atac_l4, fall back to rna_l4)
    for s in samples.values():
        if s["atac_annotation_rds"]:
            s["annotation_source"] = "atac_l4"
        elif s["rna_annotation_rds"]:
            s["annotation_source"] = "rna_l4"
        else:
            s["annotation_source"] = "none"

    # Collapse annotation lists to semicolon-separated strings + dup counts
    for s in samples.values():
        s["n_atac_annotations"] = len(s["atac_annotation_rds"])
        s["n_rna_annotations"] = len(s["rna_annotation_rds"])
        s["atac_annotation_rds"] = ";".join(s["atac_annotation_rds"])
        s["rna_annotation_rds"] = ";".join(s["rna_annotation_rds"])

    # Point every row at the combined CSVs (R script output is one CSV per modality)
    atac_csv_path = os.path.join(args.data_dir, ATAC_COMBINED_CSV)
    rna_csv_path = os.path.join(args.data_dir, RNA_COMBINED_CSV)
    for s in samples.values():
        s["atac_annotation_csv"] = atac_csv_path if s["atac_annotation_rds"] else ""
        s["rna_annotation_csv"] = rna_csv_path if s["rna_annotation_rds"] else ""

    # Write CSV
    out_path = os.path.join(args.data_dir, "sample_mapping.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = [
        "sample_id",
        "cancer_type",
        "organ",
        "diagnosis",
        "fragment_file_path",
        "matrix_dir",
        "atac_annotation_rds",
        "rna_annotation_rds",
        "n_atac_annotations",
        "n_rna_annotations",
        "atac_annotation_csv",
        "rna_annotation_csv",
        "annotation_source",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sid in sorted(samples):
            writer.writerow(samples[sid])

    print(f"Written {len(samples)} samples to {out_path}")

    # Summary
    sources = defaultdict(int)
    cancers = defaultdict(int)
    multi_atac = 0
    multi_rna = 0
    for s in samples.values():
        sources[s["annotation_source"]] += 1
        cancers[s["cancer_type"]] += 1
        if s["n_atac_annotations"] > 1:
            multi_atac += 1
        if s["n_rna_annotations"] > 1:
            multi_rna += 1
    print("\nAnnotation source:")
    for k, v in sorted(sources.items()):
        print(f"  {k}: {v}")
    print(f"\nSamples with multiple ATAC annotations: {multi_atac}")
    print(f"Samples with multiple RNA annotations: {multi_rna}")
    print("\nCancer types:")
    for k, v in sorted(cancers.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
