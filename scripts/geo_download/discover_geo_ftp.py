"""Discover multiome supplementary files on GEO FTP for given GSE accessions.

Fetches GSM-level metadata via NCBI SOFT format, then scans per-sample FTP
directories to find ATAC fragment files and RNA count matrices. Outputs a
download manifest TSV compatible with download_multiome.py.

Usage:
    python scripts/geo_download/discover_geo_ftp.py GSE295277 GSE295308 \
        --output data/gse295277_gse295308_download_manifest.tsv

    python scripts/geo_download/discover_geo_ftp.py GSE247692 --dry-run
"""

import argparse
import ftplib
import os
import re
import time
import urllib.error
import urllib.request

# ── FTP config ─────────────────────────────────────────────────────────
FTP_HOST = "ftp.ncbi.nlm.nih.gov"
GEO_SOFT_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc}&targ=gsm&form=text&view=brief"
MAX_RETRIES = 3
RETRY_DELAYS = [5, 15, 45]
FTP_RATE_LIMIT = 1.0  # seconds between FTP requests
HTTP_RATE_LIMIT = 0.5  # seconds between NCBI HTTP requests

# ── File classification patterns ───────────────────────────────────────
FRAGMENT_RE = re.compile(r"(fragment|frag).*\.tsv\.gz$", re.IGNORECASE)
FRAGMENT_INDEX_RE = re.compile(r"(fragment|frag).*\.tsv\.gz\.tbi$", re.IGNORECASE)
H5AD_RE = re.compile(r"\.h5ad(\.gz)?$", re.IGNORECASE)
H5_RE = re.compile(r"\.h5$", re.IGNORECASE)
MTX_RE = re.compile(r"(matrix\.mtx\.gz|features\.tsv\.gz|barcodes\.tsv\.gz)$", re.IGNORECASE)


def classify_file(filename):
    """Classify a supplementary file by type. Returns data_type or None."""
    if FRAGMENT_INDEX_RE.search(filename):
        return "atac_fragment_index"
    if FRAGMENT_RE.search(filename):
        return "atac_fragment"
    if H5AD_RE.search(filename):
        return "rna_h5ad"
    if H5_RE.search(filename):
        return "rna_h5"
    if MTX_RE.search(filename):
        return "rna_mtx"
    return None


# ── SOFT metadata parsing ──────────────────────────────────────────────
def fetch_gsm_metadata(gse_id):
    """Fetch GSM IDs and titles from a GSE SOFT record.

    Returns list of dicts: [{gsm_id, title, gse_id}, ...]
    """
    url = GEO_SOFT_URL.format(acc=gse_id)
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(HTTP_RATE_LIMIT)  # N2: rate limit NCBI HTTP requests
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "regularizedvi-geo-download/1.0")
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            break
        except (OSError, urllib.error.URLError) as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                print(f"  SOFT fetch failed ({e}), retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"  ERROR: could not fetch SOFT for {gse_id}: {e}")
                return []

    gsms = []
    current_gsm = None
    current_title = None
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("^SAMPLE = "):
            if current_gsm:
                gsms.append({"gsm_id": current_gsm, "title": current_title or "", "gse_id": gse_id})
            current_gsm = line.split("=")[1].strip()
            current_title = None
        elif line.startswith("!Sample_title = "):
            current_title = line.split("=", 1)[1].strip()
    if current_gsm:
        gsms.append({"gsm_id": current_gsm, "title": current_title or "", "gse_id": gse_id})

    return gsms


# ── FTP directory listing ──────────────────────────────────────────────
def acc_to_ftp_path(acc, kind="samples"):
    """Convert accession to FTP directory path.

    kind="samples" for GSM, kind="series" for GSE.
    """
    prefix = acc[:3]  # GSM or GSE
    numeric = acc[3:]
    nnn = numeric[:-3] + "nnn" if len(numeric) > 3 else "nnn"
    return f"/geo/{kind}/{prefix}{nnn}/{acc}/suppl/"


def ftp_list_files(ftp, path):
    """List files in an FTP directory. Returns [(filename, size_bytes), ...]."""
    files = []
    try:
        lines = []
        ftp.retrlines(f"LIST {path}", lines.append)
        for line in lines:
            parts = line.split()
            if len(parts) >= 9 and not parts[0].startswith("d"):
                filename = " ".join(parts[8:])
                try:
                    size = int(parts[4])
                except (ValueError, IndexError):
                    size = 0
                files.append((filename, size))
    except (ftplib.error_perm, ftplib.error_temp) as e:
        code = str(e)[:3]
        if code in ("550", "450"):
            pass  # directory not found
        else:
            print(f"  FTP error listing {path}: {e}")
    except (EOFError, OSError, ftplib.error_reply) as e:
        print(f"  FTP connection error listing {path}: {e}")
    return files


def connect_ftp_with_retry():
    """Connect to NCBI FTP server with retry logic (N3 fix)."""
    for attempt in range(MAX_RETRIES):
        try:
            ftp = ftplib.FTP(FTP_HOST, timeout=30)
            ftp.login()
            return ftp
        except (OSError, ftplib.error_temp, ftplib.error_reply) as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                print(f"  FTP connect failed ({e}), retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise RuntimeError(f"Could not connect to {FTP_HOST} after {MAX_RETRIES} attempts: {e}") from e


def reconnect_ftp(ftp):
    """Reconnect if the FTP connection dropped (N4 fix: retry on failure)."""
    try:
        ftp.voidcmd("NOOP")
        return ftp
    except (OSError, ftplib.all_errors):
        try:
            ftp.quit()
        except (OSError, ftplib.all_errors):
            pass
        return connect_ftp_with_retry()


# ── Sample ID extraction ──────────────────────────────────────────────
def extract_sample_id(gsm_title, gsm_id):
    """Extract a clean sample_id from the GSM title.

    Strategy:
    1. If title contains a comma followed by assay-type text (Chromium, ATAC,
       Gene Expression, etc.), take only the part before the first such comma.
    2. Strip remaining assay-type indicators.
    3. Clean up whitespace/punctuation.
    Falls back to the GSM ID if the title is empty or unhelpful.
    """
    if not gsm_title or gsm_title.strip() == "":
        return gsm_id

    sid = gsm_title.strip()

    # Split on comma and check if suffix is an assay descriptor.
    # Common GEO pattern: "SampleName, Chromium Single Cell ATAC, ..."
    # or "SampleName, 10x Multiome ATAC+GEX"
    assay_patterns = re.compile(
        r"^\s*(Chromium|Single Cell|10x|ATAC|Gene Expression|RNA|GEX|"
        r"scMultiome|snMultiome|Multiome|Sorting Strategy|scRNA|snRNA|"
        r"scATAC|snATAC)",
        re.IGNORECASE,
    )
    parts = sid.split(",")
    # Keep parts that don't look like assay descriptors
    clean_parts = [parts[0]]
    for part in parts[1:]:
        if not assay_patterns.match(part):
            clean_parts.append(part)
    sid = ",".join(clean_parts)

    # Remove remaining assay-type keywords
    for pat in [
        r"\bscMultiome[-_ ]?seq\b",
        r"\bsnMultiome[-_ ]?seq\b",
        r"\bmultiome\b",
        r"\b10x\s+multiome\b",
        r"\bATAC\+GEX\b",
        r"\bRNA\+ATAC\b",
        r"\bGEX\+ATAC\b",
    ]:
        sid = re.sub(pat, "", sid, flags=re.IGNORECASE)

    # Strip trailing modality suffixes so ATAC+RNA GSMs get the same sample_id
    # Handles: _ATAC, _RNA, _GEX, _RNA-Seq, _ATAC-Seq, [ATAC], [GEX], [RNA]
    sid = re.sub(r"[-_ ]*\[(ATAC|GEX|RNA)\]\s*$", "", sid, flags=re.IGNORECASE)
    sid = re.sub(r"[-_ ]+(ATAC|RNA|GEX|RNA-Seq|ATAC-Seq)\s*$", "", sid, flags=re.IGNORECASE)
    # Also handle leading modality prefixes: "RNA_preBCG_R1" → "preBCG_R1"
    sid = re.sub(r"^(ATAC|RNA|GEX|RNA-Seq|ATAC-Seq)[-_ ]+", "", sid, flags=re.IGNORECASE)

    # Clean up
    sid = sid.strip(" -_:,;")
    sid = re.sub(r"\s+", "_", sid)
    sid = re.sub(r"_+", "_", sid)
    sid = sid.strip("_")

    if not sid:
        return gsm_id
    return sid


# ── Main discovery ─────────────────────────────────────────────────────
def discover_dataset(gse_ids, sample_overrides=None):
    """Discover all supplementary files for a list of GSE IDs.

    Returns list of manifest rows (dicts).
    """
    sample_overrides = sample_overrides or {}
    all_gsms = []

    # Step 1: Fetch GSM metadata for each GSE
    for gse_id in gse_ids:
        print(f"\nFetching SOFT metadata for {gse_id}...")
        gsms = fetch_gsm_metadata(gse_id)
        print(f"  Found {len(gsms)} samples")
        for g in gsms:
            print(f"    {g['gsm_id']}: {g['title']}")
        all_gsms.extend(gsms)

    if not all_gsms:
        print("ERROR: No GSM samples found!")
        return []

    # Step 2: Scan FTP for each GSM (N1 fix: wrap in try/finally)
    print(f"\nScanning FTP for {len(all_gsms)} samples...")
    ftp = connect_ftp_with_retry()
    manifest_rows = []

    try:
        for i, gsm in enumerate(all_gsms):
            gsm_id = gsm["gsm_id"]
            gse_id = gsm["gse_id"]
            title = gsm["title"]

            # Use override if provided, else extract from title
            sample_id = sample_overrides.get(gsm_id, extract_sample_id(title, gsm_id))

            if i > 0:
                time.sleep(FTP_RATE_LIMIT)

            ftp = reconnect_ftp(ftp)
            ftp_path = acc_to_ftp_path(gsm_id, kind="samples")
            files = ftp_list_files(ftp, ftp_path)

            if not files:
                print(f"  [{i + 1}/{len(all_gsms)}] {gsm_id} ({sample_id}): no suppl files")
                continue

            classified = 0
            for filename, size_bytes in files:
                data_type = classify_file(filename)
                if data_type is None:
                    continue
                classified += 1
                url = f"ftp://{FTP_HOST}{ftp_path}{filename}"
                size_mb = round(size_bytes / (1024 * 1024), 1)
                manifest_rows.append(
                    {
                        "gsm_id": gsm_id,
                        "sample_id": sample_id,
                        "filename": filename,
                        "data_type": data_type,
                        "url": url,
                        "size_mb": size_mb,
                        "gse_id": gse_id,
                    }
                )

            total = len(files)
            print(f"  [{i + 1}/{len(all_gsms)}] {gsm_id} ({sample_id}): {total} files, {classified} classified")

        # Step 3: Also check series-level supplementary files
        for gse_id in gse_ids:
            time.sleep(FTP_RATE_LIMIT)
            ftp = reconnect_ftp(ftp)
            ftp_path = acc_to_ftp_path(gse_id, kind="series")
            files = ftp_list_files(ftp, ftp_path)
            if files:
                print(f"\n  Series-level files for {gse_id}: {len(files)} files")
                for filename, size_bytes in files:
                    data_type = classify_file(filename)
                    if data_type is not None:
                        print(f"    {filename} ({data_type}, {size_bytes / 1024 / 1024:.1f} MB)")
                        # Series-level files use GSE as sample_id (need manual assignment)
                        url = f"ftp://{FTP_HOST}{ftp_path}{filename}"
                        size_mb = round(size_bytes / (1024 * 1024), 1)
                        manifest_rows.append(
                            {
                                "gsm_id": gse_id,
                                "sample_id": gse_id,
                                "filename": filename,
                                "data_type": data_type,
                                "url": url,
                                "size_mb": size_mb,
                                "gse_id": gse_id,
                            }
                        )
    finally:
        # N1 fix: always close FTP connection
        try:
            ftp.quit()
        except (OSError, ftplib.all_errors):
            pass

    return manifest_rows


def write_manifest(rows, output_path):
    """Write manifest rows to TSV file."""
    abs_output = os.path.abspath(output_path)  # F1 fix: consistent absolute path
    fieldnames = ["gsm_id", "sample_id", "filename", "data_type", "url", "size_mb", "gse_id"]
    out_dir = os.path.dirname(abs_output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(abs_output, "w") as f:
        f.write("\t".join(fieldnames) + "\n")
        for row in rows:
            f.write("\t".join(str(row[k]) for k in fieldnames) + "\n")
    print(f"\nWritten {len(rows)} rows to {abs_output}")


def print_summary(rows):
    """Print a summary table of discovered files."""
    print("\n" + "=" * 70)
    print("DISCOVERY SUMMARY")
    print("=" * 70)

    # By data type
    type_counts = {}
    type_sizes = {}
    for r in rows:
        dt = r["data_type"]
        type_counts[dt] = type_counts.get(dt, 0) + 1
        type_sizes[dt] = type_sizes.get(dt, 0) + r["size_mb"]

    print(f"\n{'Data Type':<25} {'Count':>6} {'Size (GB)':>10}")
    print("-" * 45)
    total_size = 0
    for dt in sorted(type_counts):
        size_gb = type_sizes[dt] / 1024
        total_size += type_sizes[dt]
        print(f"{dt:<25} {type_counts[dt]:>6} {size_gb:>10.1f}")
    print("-" * 45)
    print(f"{'TOTAL':<25} {len(rows):>6} {total_size / 1024:>10.1f}")

    # By GSE
    gse_counts = {}
    for r in rows:
        gse_counts[r["gse_id"]] = gse_counts.get(r["gse_id"], 0) + 1
    if len(gse_counts) > 1:
        print("\nBy GSE:")
        for gse, count in sorted(gse_counts.items()):
            print(f"  {gse}: {count} files")

    # Unique samples
    samples = {r["sample_id"] for r in rows}
    print(f"\nUnique samples: {len(samples)}")

    # Fragment file check
    frag_samples = {r["sample_id"] for r in rows if r["data_type"] == "atac_fragment"}
    rna_samples = {r["sample_id"] for r in rows if r["data_type"] in ("rna_h5ad", "rna_h5", "rna_mtx")}
    both = frag_samples & rna_samples
    frag_only = frag_samples - rna_samples
    rna_only = rna_samples - frag_samples

    print(f"\nSamples with both RNA + ATAC fragments: {len(both)}")
    if frag_only:
        print(f"Samples with ATAC only: {len(frag_only)} — {sorted(frag_only)}")
    if rna_only:
        print(f"Samples with RNA only: {len(rna_only)} — {sorted(rna_only)}")

    print("=" * 70)


def load_sample_overrides(path):
    """Load sample ID overrides from a JSON file. Format: {GSM_ID: sample_id}."""
    import json

    with open(path) as f:
        return json.load(f)


def main():
    """Discover multiome files on GEO FTP and generate a download manifest."""
    parser = argparse.ArgumentParser(description="Discover multiome supplementary files on GEO FTP")
    parser.add_argument("gse_ids", nargs="+", help="GSE accession(s) to scan")
    parser.add_argument("--output", "-o", required=True, help="Output manifest TSV path")
    parser.add_argument("--dry-run", action="store_true", help="Show results without writing")
    parser.add_argument(
        "--sample-map",
        default=None,
        help="JSON file mapping GSM IDs to explicit sample_ids",
    )
    args = parser.parse_args()

    overrides = {}
    if args.sample_map:
        overrides = load_sample_overrides(args.sample_map)
        print(f"Loaded {len(overrides)} sample ID overrides from {args.sample_map}")

    rows = discover_dataset(args.gse_ids, sample_overrides=overrides)
    print_summary(rows)

    if args.dry_run:
        print("\n[DRY RUN] No manifest written.")
    else:
        write_manifest(rows, args.output)


if __name__ == "__main__":
    main()
