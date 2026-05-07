"""Download HDMA (Human Development Multiomic Atlas) data from Zenodo.

Reads a TSV of Zenodo record IDs, queries the Zenodo API for each
record's file listing, maps each file to a canonical local path,
and downloads only the files we want with `wget -c` (resumable).

Mirrors the structure of scripts/geo_download/download_multiome.py
but for Zenodo records instead of GEO FTP, and with HDMA-specific
filename parsing.

File-type detection is regex-based on filename:
  *.fragments.tsv.gz       -> atac/<sample_id>/fragments.tsv.gz
  *.fragments.tsv.gz.tbi   -> atac/<sample_id>/fragments.tsv.gz.tbi
  *.barcodes.tsv.gz        -> rna/<sample_id>/barcodes.tsv.gz
  *.features.tsv.gz        -> rna/<sample_id>/features.tsv.gz
  *.matrix.mtx.gz          -> rna/<sample_id>/matrix.mtx.gz
  per_cell_meta.csv        -> annotations/per_cell_meta.csv
  hdma_global_caCREs.bed   -> caCREs/hdma_global_caCREs.bed
  all_training_regions.tar.gz -> caCREs/all_training_regions.tar.gz

Sample IDs follow the published convention T<donor>_b<batch>_<organ>_PCW<age>
(e.g. T014_b11_Heart_PCW18). The sample-id prefix is stripped on rename.

Usage:
    python scripts/zenodo_download/download_hdma.py \
        --records scripts/zenodo_download/zenodo_records.tsv \
        --output-dir /nemo/lab/briscoej/home/users/kleshcv/large_data/HDMA \
        --dry-run

    python scripts/zenodo_download/download_hdma.py \
        --records scripts/zenodo_download/zenodo_records.tsv \
        --output-dir /nemo/lab/briscoej/home/users/kleshcv/large_data/HDMA
"""

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ZENODO_API = "https://zenodo.org/api/records/{record_id}"

SAMPLE_FILE_RE = re.compile(
    r"^(?P<sample>T\w+?_b\d+_[A-Za-z]+_PCW\w+)\.(?P<role>"
    r"fragments\.tsv\.gz\.tbi|fragments\.tsv\.gz|"
    r"barcodes\.tsv\.gz|features\.tsv\.gz|matrix\.mtx\.gz)$"
)

ROLE_TO_LOCAL = {
    "fragments.tsv.gz": ("atac", "fragments.tsv.gz"),
    "fragments.tsv.gz.tbi": ("atac", "fragments.tsv.gz.tbi"),
    "barcodes.tsv.gz": ("rna", "barcodes.tsv.gz"),
    "features.tsv.gz": ("rna", "features.tsv.gz"),
    "matrix.mtx.gz": ("rna", "matrix.mtx.gz"),
}

# Files we want from the combined annotations record (record 17427146).
# Anything not listed here is skipped (motif_instances.tar.gz, abc_*.zip, etc.
# are deferred to the cluster-level benchmark plan).
ANNOTATIONS_KEEP = {
    "per_cell_meta.csv": ("annotations", "per_cell_meta.csv"),
    "hdma_global_caCREs.bed": ("caCREs", "hdma_global_caCREs.bed"),
    "all_training_regions.tar.gz": ("caCREs", "all_training_regions.tar.gz"),
}


def parse_records_tsv(path):
    """Parse the records TSV into a list of {column: value} dicts."""
    rows = []
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            vals = line.rstrip("\n").split("\t")
            if len(vals) == len(header):
                rows.append(dict(zip(header, vals, strict=False)))
    return rows


def fetch_zenodo_record(record_id, retries=3):
    """Fetch the Zenodo record JSON (with file listing), retrying on transient errors."""
    url = ZENODO_API.format(record_id=record_id)
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"Zenodo API failed for {record_id}: {last_err}")


def file_url(record_id, key):
    """Build the public download URL for a file inside a Zenodo record."""
    return f"https://zenodo.org/records/{record_id}/files/{key}"


def classify_sample_file(key):
    """Parse an HDMA per-sample filename into (modality, sample_id, canonical name) or None."""
    m = SAMPLE_FILE_RE.match(key)
    if not m:
        return None
    sample = m.group("sample")
    role = m.group("role")
    modality, canonical = ROLE_TO_LOCAL[role]
    return {
        "modality": modality,
        "sample_id": sample,
        "canonical": canonical,
    }


def plan_downloads(records_path, output_dir):
    """Walk every Zenodo record in the TSV and build the (url -> local path) download plan."""
    rows = parse_records_tsv(records_path)
    plan = []
    for row in rows:
        record_id = row["record_id"]
        dataset_type = row["dataset_type"]
        organ = row["organ"]

        print(f"\n[record {record_id}] {row['title']}", file=sys.stderr)
        rec = fetch_zenodo_record(record_id)
        files = rec.get("files", [])
        print(f"  {len(files)} files in record", file=sys.stderr)

        for f in files:
            key = f["key"]
            size = f.get("size", 0)
            url = file_url(record_id, key)

            if dataset_type == "fragments_mtx":
                info = classify_sample_file(key)
                if info is None:
                    print(f"  SKIP (unrecognised name): {key}", file=sys.stderr)
                    continue
                local_dir = os.path.join(output_dir, info["modality"], info["sample_id"])
                local_path = os.path.join(local_dir, info["canonical"])
                plan.append(
                    {
                        "record_id": record_id,
                        "organ": organ,
                        "sample_id": info["sample_id"],
                        "modality": info["modality"],
                        "key": key,
                        "url": url,
                        "size": size,
                        "local_path": local_path,
                    }
                )
            elif dataset_type == "annotations_caCRE":
                if key not in ANNOTATIONS_KEEP:
                    print(f"  SKIP (deferred to benchmark plan): {key}", file=sys.stderr)
                    continue
                subdir, canonical = ANNOTATIONS_KEEP[key]
                local_dir = os.path.join(output_dir, subdir)
                local_path = os.path.join(local_dir, canonical)
                plan.append(
                    {
                        "record_id": record_id,
                        "organ": organ,
                        "sample_id": "",
                        "modality": subdir,
                        "key": key,
                        "url": url,
                        "size": size,
                        "local_path": local_path,
                    }
                )
            else:
                print(f"  WARNING: unknown dataset_type '{dataset_type}'", file=sys.stderr)
    return plan


_PRINT_LOCK = threading.Lock()


def _log(msg):
    """Thread-safe stdout flush so workers' progress is interleaved cleanly."""
    with _PRINT_LOCK:
        print(msg, flush=True)


def _fetch_one(idx, total, p):
    """Download a single file with `wget -c` and built-in retries.

    Returns (p, error_or_None). Verifies post-download size matches the
    Zenodo-reported size before declaring success — otherwise treated as failure
    (Zenodo's 429/503 responses still produce wget exit 0 in some edge cases).
    """
    os.makedirs(os.path.dirname(p["local_path"]), exist_ok=True)
    _log(f"[{idx}/{total}] START  {p['key']}  ({p['size'] / 1e6:.1f} MB)")
    t0 = time.time()
    try:
        subprocess.run(
            [
                "wget",
                "-c",
                "-q",
                "--tries",
                "5",
                "--retry-connrefused",
                "--waitretry",
                "30",
                "--timeout",
                "60",
                "-O",
                p["local_path"],
                p["url"],
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        _log(f"[{idx}/{total}] FAIL   {p['key']}  (wget exit {e.returncode})")
        return (p, f"wget exit {e.returncode}")

    dt = time.time() - t0
    actual = os.path.getsize(p["local_path"]) if os.path.exists(p["local_path"]) else 0
    if p["size"] and actual != p["size"]:
        _log(f"[{idx}/{total}] PARTIAL {p['key']}  (got {actual} B, expected {p['size']} B)")
        return (p, f"size mismatch {actual}/{p['size']}")
    rate = (actual / 1e6) / dt if dt > 0 else float("inf")
    _log(f"[{idx}/{total}] OK     {p['key']}  ({dt:.1f}s, {rate:.1f} MB/s)")
    return (p, None)


def download(plan, dry_run=False, workers=8):
    """Execute the download plan with a thread pool of resumable wget workers."""
    total_size = sum(p["size"] for p in plan)
    print(f"\nPlanned files: {len(plan)}  (~{total_size / 1e9:.1f} GB total)", flush=True)

    to_fetch = []
    already = 0
    deleted_zero = 0
    for p in plan:
        if os.path.exists(p["local_path"]):
            sz = os.path.getsize(p["local_path"])
            if sz == 0:
                # Empty placeholder from a prior failed wget — remove so -c can refetch.
                os.remove(p["local_path"])
                deleted_zero += 1
                to_fetch.append(p)
            elif p["size"] and sz < p["size"]:
                # Partial — wget -c will resume.
                to_fetch.append(p)
            else:
                already += 1
        else:
            to_fetch.append(p)
    if deleted_zero:
        print(f"Removed {deleted_zero} zero-byte placeholder file(s)", flush=True)

    print(f"Already present: {already}", flush=True)
    print(f"To download:     {len(to_fetch)} (workers={workers})", flush=True)

    if dry_run:
        print("\n[DRY RUN]", flush=True)
        for p in to_fetch[:25]:
            print(f"  {p['url']}\n    -> {p['local_path']}  ({p['size'] / 1e6:.1f} MB)", flush=True)
        if len(to_fetch) > 25:
            print(f"  ... and {len(to_fetch) - 25} more", flush=True)
        return

    if not to_fetch:
        print("Nothing to download.", flush=True)
        return

    # Sort smallest-first so quick wins surface and slow giants run in parallel.
    to_fetch.sort(key=lambda p: p["size"])

    failures = []
    total = len(to_fetch)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, i, total, p): p for i, p in enumerate(to_fetch, 1)}
        for fut in concurrent.futures.as_completed(futures):
            _, err = fut.result()
            if err is not None:
                failures.append((futures[fut], err))

    print(f"\nDone. Downloaded: {total - len(failures)}, failed: {len(failures)}", flush=True)
    if failures:
        for p, err in failures:
            print(f"  FAIL  {p['key']}: {err}", flush=True)
        # Non-zero exit so the surrounding sbatch wrap (set -e) skips the
        # downstream tar extract step on any incomplete download.
        sys.exit(1)


def main():
    """CLI entry point: parse args, plan downloads, and run them."""
    parser = argparse.ArgumentParser(description="Download HDMA data from Zenodo")
    parser.add_argument("--records", required=True, help="TSV listing Zenodo record IDs")
    parser.add_argument("--output-dir", required=True, help="Output root (e.g. .../HDMA)")
    parser.add_argument("--dry-run", action="store_true", help="Show plan only")
    parser.add_argument("--workers", type=int, default=8, help="Parallel wget workers (default 8)")
    args = parser.parse_args()

    plan = plan_downloads(args.records, args.output_dir)
    download(plan, dry_run=args.dry_run, workers=args.workers)


if __name__ == "__main__":
    main()
