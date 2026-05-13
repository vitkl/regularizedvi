r"""Build the Hickey/Becker intestine multiome download manifest from the two Dryad deposits.

Pulls file lists via the public Dryad metadata API (no auth needed for browsing),
filters to multiome donors (B006, B008, B009, B010, B011, B012) plus shared
annotation/metadata files, and writes a TSV consumed by download_dryad.py.

Source: scRNA  10.5061/dryad.8pk0p2ns8  (Becker et al. 2023 — Cell Ranger matrices + clustered Seurat .rds)
        scATAC 10.5061/dryad.0zpc8672f  (atac_fragments.tsv.gz + .tbi + per-cell ATAC cell-type TSVs)

Usage:
    python scripts/intestine_hickey/build_manifest.py \
        --output data/dryad_hickey_intestine_multiome_manifest.tsv
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

# Multiome donors (B006, B008, B009, B010, B011, B012). Non-multiome B001/B004/B005 excluded.
MULTIOME_DONOR_RE = re.compile(r"^B(006|008|009|010|011|012)-A-")

# Shared (non-per-sample) files we always want from each Dryad deposit.
SHARED_FILES = {
    "scRNA": {
        # Clustered Seurat .rds objects (author cell-type labels live in seu@meta.data)
        "clustered_immune_object.rds",
        "clustered_stromal_object.rds",
        "clustered_duodenum_object.rds",
        "clustered_jejunum_object.rds",
        "clustered_ileum_object.rds",
        "clustered_colon_object.rds",
        "clustered_enteroendocrine_object.rds",
        "clustered_secretory_special_object.rds",
        # UMAP + annotations TSVs (8 files; pattern: end with _umap_and_annotations.tsv or similar)
        # → captured below by glob suffix match.
        "sample_location_metadata.csv",
        "README.md",
    },
    "scATAC": {
        # Per-cell ATAC cell-type TSVs (multiome subset only)
        "scATAC_multiome_cell_types_epithelial_colon.tsv",
        "scATAC_multiome_cell_types_epithelial_duodenum.tsv",
        "scATAC_multiome_cell_types_epithelial_ileum.tsv",
        "scATAC_multiome_cell_types_epithelial_jejunum.tsv",
        "scATAC_multiome_cell_types_immune.tsv",
        "scATAC_multiome_cell_types_stromal.tsv",
        "atac_sample_location_metadata.csv",
        "peak_matrix_metadata.csv",
        "README.md",
    },
}

# Suffixes used to also capture per-compartment UMAP+annotation TSVs in the scRNA deposit.
SHARED_SUFFIXES_RNA = (".tsv",)  # filtered further below to only TSVs with no sample prefix

# Dryad version IDs (resolved interactively via /api/v2/datasets/.../versions earlier).
DRYAD_DEPOSITS = [
    {
        "modality": "scRNA",
        "doi": "10.5061/dryad.8pk0p2ns8",
        "version_id": 321675,
    },
    {
        "modality": "scATAC",
        "doi": "10.5061/dryad.0zpc8672f",
        "version_id": 226604,
    },
]

DRYAD_API_BASE = "https://datadryad.org"


def fetch_all_files(version_id: int) -> list[dict]:
    """Paginate /api/v2/versions/<id>/files; return list of file dicts."""
    files: list[dict] = []
    page = 1
    while True:
        url = f"{DRYAD_API_BASE}/api/v2/versions/{version_id}/files?per_page=200&page={page}"
        with urllib.request.urlopen(url, timeout=60) as r:
            payload = json.load(r)
        chunk = payload.get("_embedded", {}).get("stash:files", [])
        files.extend(chunk)
        # _links.next absence or empty chunk → done
        next_link = payload.get("_links", {}).get("next")
        if not next_link or not chunk:
            break
        page += 1
        if page > 50:  # safety
            break
    return files


def classify(path: str, modality: str) -> tuple[str | None, str | None, str | None]:
    """Return (data_type, sample_id, donor) for a file path; or (None,None,None) to skip."""
    name = path

    # Per-sample matches
    m = MULTIOME_DONOR_RE.match(name)
    if m:
        # Sample prefix is everything up to the first '_' that follows the region tag.
        # e.g. "B006-A-001_atac_fragments.tsv.gz" → sample "B006-A-001", donor "B006"
        #      "B006-A-201-R2_atac_fragments.tsv.gz" → sample "B006-A-201-R2"
        sample = name.split("_", 1)[0]
        donor = f"B{m.group(1)}"
        # data_type from suffix
        if name.endswith("_atac_fragments.tsv.gz"):
            return ("atac_fragment", sample, donor)
        if name.endswith("_atac_fragments.tsv.gz.tbi"):
            return ("atac_fragment_index", sample, donor)
        if name.endswith("_barcodes.tsv.gz"):
            return ("rna_mtx", sample, donor)
        if name.endswith("_features.tsv.gz"):
            return ("rna_mtx", sample, donor)
        if name.endswith("_matrix.mtx.gz"):
            return ("rna_mtx", sample, donor)
        return (None, None, None)  # unrecognised per-sample suffix → skip

    # Shared files
    shared = SHARED_FILES.get(modality, set())
    if name in shared:
        return ("annotation", "META", "META")

    # Additional shared TSVs from scRNA deposit (compartment UMAP/annotation tables).
    if (
        modality == "scRNA"
        and name.endswith(".tsv")
        and not MULTIOME_DONOR_RE.search(name)
        and not name.startswith(("B001", "B004", "B005"))
    ):
        return ("annotation", "META", "META")

    return (None, None, None)


def parse_sample(sample: str) -> tuple[str, str]:
    """Sample 'B006-A-201-R2' → (donor='B006', region_code='A-201-R2')."""
    parts = sample.split("-", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def file_to_row(file_obj: dict, modality: str, doi: str) -> dict | None:
    """Convert a Dryad file metadata dict into a manifest row, or None to skip."""
    path = file_obj.get("path", "")
    size = file_obj.get("size", 0)
    file_id = (file_obj.get("_links", {}).get("self", {}).get("href", "") or "").rsplit("/", 1)[-1]
    download_href = file_obj.get("_links", {}).get("stash:download", {}).get("href", "")
    data_type, sample, donor = classify(path, modality)
    if data_type is None:
        return None
    # Hickey is not from GEO; reuse the gsm_id column as a generic ID for the existing downloader.
    return {
        "gsm_id": sample,
        "sample_id": sample,
        "donor": donor,
        "modality": modality,
        "filename": path,
        "data_type": data_type,
        "url": f"{DRYAD_API_BASE}{download_href}"
        if download_href
        else f"{DRYAD_API_BASE}/api/v2/files/{file_id}/download",
        "size_mb": f"{size / 1_000_000:.2f}" if isinstance(size, int) else "",
        "size_bytes": str(size) if isinstance(size, int) else "",
        "file_id": file_id,
        "accession": doi,
    }


def main():
    """Entry point: fetch Dryad file lists, filter, write the manifest TSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Output TSV path")
    args = parser.parse_args()

    rows: list[dict] = []
    for deposit in DRYAD_DEPOSITS:
        modality = deposit["modality"]
        doi = deposit["doi"]
        version_id = deposit["version_id"]
        print(f"\n[{modality}] fetching file list from version {version_id} (DOI {doi})...")
        files = fetch_all_files(version_id)
        print(f"  total files in deposit: {len(files)}")
        matched = 0
        for f in files:
            row = file_to_row(f, modality, doi)
            if row is not None:
                rows.append(row)
                matched += 1
        print(f"  matched (multiome + shared): {matched}")

    # Sanity: count per-donor sample files
    by_donor: dict[str, int] = {}
    for r in rows:
        if r["sample_id"] != "META":
            by_donor[r["donor"]] = by_donor.get(r["donor"], 0) + 1
    print("\nPer-donor file counts (multiome subset):")
    for d in sorted(by_donor):
        print(f"  {d}: {by_donor[d]} files")
    print(f"\nTotal rows: {len(rows)}")

    columns = [
        "gsm_id",
        "sample_id",
        "donor",
        "modality",
        "filename",
        "data_type",
        "url",
        "size_mb",
        "size_bytes",
        "file_id",
        "accession",
    ]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("\t".join(columns) + "\n")
        for r in rows:
            f.write("\t".join(r.get(c, "") for c in columns) + "\n")
    print(f"\nWrote {len(rows)} rows → {out}")
    total_gb = sum(int(r["size_bytes"] or 0) for r in rows) / 1e9
    print(f"Total payload: {total_gb:.1f} GB")


if __name__ == "__main__":
    main()
