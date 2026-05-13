r"""Download files from Dryad with OAuth2 bearer-token auth (required for programmatic file access).

Dryad's per-file URLs `/downloads/file_stream/<id>` are behind an AWS WAF JS challenge
that blocks bare curl/wget. The supported programmatic path is the API endpoint
`/api/v2/files/<id>/download` with an `Authorization: Bearer <token>` header.

Credentials: ~/.dryad_credentials (INI; created once via the Dryad web UI):

    [dryad]
    app_id  = <your_app_id>
    secret  = <your_secret>

Manifest TSV columns (compatible with the existing scripts/geo_download/ flow):
    gsm_id, sample_id, donor, modality, filename, data_type, url,
    size_mb, size_bytes, file_id, accession

Layout (matches existing immune datasets — per-sample subdir, canonical filename
except for rna_mtx and annotation which preserve the original filename):
    rna/{sample_id}/<original_or_canonical>
    atac/{sample_id}/<original_or_canonical>
    annotations/<original>

Usage:
    python scripts/intestine_hickey/download_dryad.py \
        --manifest data/dryad_hickey_intestine_multiome_manifest.tsv \
        --output-dir /nemo/lab/briscoej/home/users/kleshcv/large_data/intestine_hickey \
        [--credentials ~/.dryad_credentials] [--dry-run] [--workers 4]
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DRYAD_BASE = "https://datadryad.org"
TOKEN_URL = f"{DRYAD_BASE}/oauth/token"
TOKEN_LIFETIME_S = 10 * 3600  # documented Dryad token lifetime
TOKEN_REFRESH_MARGIN_S = 30 * 60  # refresh if older than (10h - 30 min) = 9.5h

# Mirror the SUBDIR_MAP / CANONICAL_NAME conventions from
# scripts/geo_download/download_multiome.py (kept in sync intentionally).
SUBDIR_MAP = {
    "rna_h5ad": "rna/{sample_id}",
    "rna_h5": "rna/{sample_id}",
    "rna_mtx": "rna/{sample_id}",
    "atac_fragment": "atac/{sample_id}",
    "atac_fragment_index": "atac/{sample_id}",
    "annotation": "annotations",
}
CANONICAL_NAME = {
    "rna_h5ad": "adata.h5ad.gz",
    "rna_h5": "filtered_feature_bc_matrix.h5",
    "rna_mtx": "{original}",
    "atac_fragment": "atac_fragments.tsv.gz",
    "atac_fragment_index": "atac_fragments.tsv.gz.tbi",
    "annotation": "{original}",
}


def load_credentials(path: Path) -> tuple[str, str]:
    """Read [dryad] app_id + secret from ~/.dryad_credentials (chmod 600 expected)."""
    if not path.exists():
        sys.exit(
            f"ERROR: credentials file {path} not found. Create it with:\n"
            f"  cat > {path} <<EOF\n"
            f"  [dryad]\n  app_id = <YOUR_APP_ID>\n  secret = <YOUR_SECRET>\n  EOF\n"
            f"  chmod 600 {path}\n"
            f"Dryad API account setup: https://datadryad.org/ → ORCID login → "
            f"My account → API accounts → Create."
        )
    cfg = configparser.ConfigParser()
    cfg.read(path)
    if "dryad" not in cfg.sections():
        sys.exit(f"ERROR: [dryad] section missing in {path}")
    app_id = cfg["dryad"].get("app_id", "").strip()
    secret = cfg["dryad"].get("secret", "").strip()
    if not app_id or not secret:
        sys.exit(f"ERROR: app_id or secret empty in {path}")
    return app_id, secret


def mint_token(app_id: str, secret: str) -> str:
    """OAuth2 client-credentials grant → 10-hour bearer token."""
    data = urllib.parse.urlencode(
        {
            "grant_type": "client_credentials",
            "client_id": app_id,
            "client_secret": secret,
        }
    ).encode()
    req = urllib.request.Request(
        TOKEN_URL,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        payload = json.load(r)
    tok = payload.get("access_token")
    if not tok:
        sys.exit(f"ERROR: no access_token in OAuth response: {payload}")
    return tok


class TokenManager:
    """Holds a bearer token; refreshes when older than ~9.5h. Thread-safe."""

    def __init__(self, app_id: str, secret: str):
        self._app_id = app_id
        self._secret = secret
        self._token: str | None = None
        self._minted_at: float = 0.0
        self._lock = threading.Lock()

    def get(self) -> str:
        """Return a valid bearer token, minting/refreshing under the lock if needed."""
        with self._lock:
            if self._token is None or (time.time() - self._minted_at) > (TOKEN_LIFETIME_S - TOKEN_REFRESH_MARGIN_S):
                self._token = mint_token(self._app_id, self._secret)
                self._minted_at = time.time()
                print(f"  [token] minted at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
            return self._token

    def invalidate(self) -> None:
        """Drop the cached token so the next get() re-mints. Used after a 401."""
        with self._lock:
            self._token = None


def parse_manifest(path: Path) -> list[dict]:
    """Read a tab-separated manifest into a list of header-keyed dict rows."""
    rows: list[dict] = []
    with path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            vals = line.rstrip("\n").split("\t")
            if len(vals) == len(header):
                rows.append(dict(zip(header, vals, strict=False)))
    return rows


def local_path(row: dict, output_dir: Path) -> Path | None:
    """Resolve the destination Path for a manifest row, or None for unknown data_type."""
    data_type = row["data_type"]
    sample_id = row["sample_id"]
    sub = SUBDIR_MAP.get(data_type)
    name = CANONICAL_NAME.get(data_type)
    if sub is None or name is None:
        print(f"  WARNING: unknown data_type '{data_type}'; skipping {row.get('filename', '?')}")
        return None
    sub = sub.format(sample_id=sample_id)
    name = name.format(original=row["filename"]) if "{original}" in name else name
    return output_dir / sub / name


def already_downloaded(row: dict, dest: Path) -> bool:
    """Skip if a file exists with the expected size."""
    if not dest.exists():
        return False
    expected = int(row.get("size_bytes") or 0)
    if expected <= 0:
        return True  # unknown expected size → trust existence
    actual = dest.stat().st_size
    return actual == expected


def download_one(row: dict, dest: Path, token_manager: TokenManager, attempt: int = 1) -> tuple[dict, str | None]:
    """Download a single file via the bearer-token API endpoint. Returns (row, err or None)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    url = row["url"]
    headers = {
        "Authorization": f"Bearer {token_manager.get()}",
        "User-Agent": "regularizedvi-dryad-downloader/1.0",
    }
    start_byte = 0
    if tmp.exists():
        start_byte = tmp.stat().st_size
        headers["Range"] = f"bytes={start_byte}-"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            # If we asked for a Range but the server returned the full body (200 vs 206),
            # discard the prior partial and start from scratch.
            if start_byte > 0 and resp.status != 206:
                tmp.unlink(missing_ok=True)
                start_byte = 0
            mode = "ab" if start_byte else "wb"
            with tmp.open(mode) as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
        expected = int(row.get("size_bytes") or 0)
        actual = tmp.stat().st_size
        if expected > 0 and actual != expected:
            tmp.unlink(missing_ok=True)
            return (row, f"size mismatch: got {actual}, expected {expected} (cleaned up .part)")
        os.replace(tmp, dest)
        return (row, None)
    except urllib.error.HTTPError as e:
        if e.code == 401 and attempt == 1:
            token_manager.invalidate()
            time.sleep(1)
            return download_one(row, dest, token_manager, attempt=2)
        return (row, f"HTTP {e.code} {e.reason}")
    except Exception as e:  # noqa: BLE001
        return (row, str(e))


def main():
    """Entry point: plan downloads, mint a token, fetch files in parallel, report failures."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--credentials", type=Path, default=Path.home() / ".dryad_credentials")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=4, help="Parallel downloads (Dryad recommends ≤ 4)")
    ap.add_argument("--limit", type=int, default=0, help="Cap total downloads (for testing)")
    args = ap.parse_args()

    rows = parse_manifest(args.manifest)
    print(f"Manifest: {len(rows)} rows from {args.manifest}")
    print(f"Output:   {args.output_dir}")

    # Plan
    plan: list[tuple[dict, Path]] = []
    skipped = 0
    skipped_unknown = 0
    for row in rows:
        dest = local_path(row, args.output_dir)
        if dest is None:
            skipped_unknown += 1
            continue
        if already_downloaded(row, dest):
            skipped += 1
            continue
        plan.append((row, dest))

    total_gb = sum(int(r["size_bytes"] or 0) for r, _ in plan) / 1e9
    print(f"  to download:   {len(plan)}  ({total_gb:.1f} GB)")
    print(f"  already on disk: {skipped}")
    print(f"  unknown type:    {skipped_unknown}")
    if args.dry_run:
        for row, dest in plan[: args.limit or len(plan)]:
            print(f"  {row['filename']:60s} -> {dest}")
        return

    if not plan:
        print("Nothing to download.")
        return

    if args.limit:
        plan = plan[: args.limit]
        print(f"  --limit {args.limit}: capping to {len(plan)} downloads")

    # Auth: mint initial token
    app_id, secret = load_credentials(args.credentials)
    tm = TokenManager(app_id, secret)
    tm.get()  # mint up-front

    failed: list[tuple[dict, str]] = []
    completed = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(download_one, row, dest, tm): (row, dest) for row, dest in plan}
        for fut in as_completed(futures):
            row, err = fut.result()
            completed += 1
            if err is None:
                size_mb = (int(row.get("size_bytes") or 0)) / 1e6
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                print(
                    f"  [{completed}/{len(plan)}] OK  {row['filename']}  ({size_mb:.1f} MB)  [avg {rate:.2f} files/s]",
                    flush=True,
                )
            else:
                failed.append((row, err))
                print(f"  [{completed}/{len(plan)}] FAIL {row['filename']}: {err}", flush=True)

    print(f"\nDone. Downloaded: {len(plan) - len(failed)}, Failed: {len(failed)}")
    if failed:
        print("\nFailures (rerun the same command to retry — script is idempotent):")
        for row, err in failed:
            print(f"  {row['filename']}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
