"""Inspect GBM-Space 59 GB h5ad via h5py only (no anndata.read_h5ad).

Phase 0 step 0.3 of immune_integration_v2 (see [plan](../../.claude/plans/implement-these-steps-in-tranquil-parasol.md)).

The h5ad is too large for `sc.read_h5ad` in memory (~ 59 GB on disk; sparse expanded would OOM).
h5py reads obs categoricals + a small slice of /X/data with RSS < 2 GB in seconds.

Output: stdout dump of /X group layout (CSR vs dense), dtype/shape/first-100 values of /X/data,
and value_counts of obs/annotation_granular, obs/annotation_coarse, obs/sample, obs/donor_id.
All 155 samples are inspected (no 118-subset filter per grill resolution).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import h5py

DEFAULT_H5AD = "/nemo/lab/briscoej/home/users/kleshcv/large_data/gbm/GBM_space_snRNA.h5ad"
OBS_COLS = ("annotation_granular", "annotation_coarse", "sample", "donor_id")


def parse_args() -> argparse.Namespace:
    """Parse CLI args: --h5ad, --n-data-preview."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--h5ad", default=DEFAULT_H5AD, help="Path to GBM_space_snRNA.h5ad")
    p.add_argument("--n-data-preview", type=int, default=100, help="Number of /X/data values to preview")
    return p.parse_args()


def dump_obs_value_counts(f: h5py.File, col: str) -> None:
    """Print value_counts for an obs column (categorical or raw) to stdout."""
    if col not in f["obs"]:
        print(f"  obs/{col}: NOT PRESENT", flush=True)
        return
    node = f["obs"][col]
    if "categories" in node and "codes" in node:
        cats = [c.decode() if isinstance(c, bytes) else c for c in node["categories"][:]]
        codes = node["codes"][:]
        counts = Counter(cats[c] if c >= 0 else "<NA>" for c in codes)
        print(f"  obs/{col} (categorical): {len(counts)} unique, {sum(counts.values()):,} cells", flush=True)
        for k, v in counts.most_common():
            print(f"    {k!r:50s}  {v:>10,d}", flush=True)
    else:
        vals = node[:]
        decoded = [v.decode() if isinstance(v, bytes) else v for v in vals]
        counts = Counter(decoded)
        print(f"  obs/{col} (raw): {len(counts)} unique, {sum(counts.values()):,} cells", flush=True)
        for k, v in counts.most_common(40):
            print(f"    {k!r:50s}  {v:>10,d}", flush=True)
        if len(counts) > 40:
            print(f"    ... ({len(counts) - 40} more)", flush=True)


def dump_x_layout(f: h5py.File, n_data_preview: int) -> None:
    """Print /X layout (CSR sparse vs dense) + dtype/shape + first-N values to stdout."""
    if "X" not in f:
        print("  /X: NOT PRESENT", flush=True)
        return
    x = f["X"]
    if isinstance(x, h5py.Group):
        members = {k: type(x[k]).__name__ for k in x.keys()}
        print(f"  /X: Group with members {members}", flush=True)
        if "data" in x and "indices" in x and "indptr" in x:
            data = x["data"]
            print(f"    /X/data: dtype={data.dtype}  shape={data.shape}  size={data.shape[0]:,}", flush=True)
            print(f"    /X/indices: dtype={x['indices'].dtype}  shape={x['indices'].shape}", flush=True)
            print(f"    /X/indptr: dtype={x['indptr'].dtype}  shape={x['indptr'].shape}", flush=True)
            print("    layout: CSR/CSC sparse", flush=True)
            preview = data[: min(n_data_preview, data.shape[0])]
            print(f"    /X/data first {len(preview)} values: {preview.tolist()}", flush=True)
    else:
        print(f"  /X: Dataset dtype={x.dtype}  shape={x.shape}  (dense)", flush=True)
        preview = x[0, : min(n_data_preview, x.shape[1])]
        print(f"    /X[0, :{len(preview)}] preview: {preview.tolist()}", flush=True)


def main() -> int:
    """Open h5ad with h5py and dump /X layout + obs value_counts for OBS_COLS."""
    args = parse_args()
    p = Path(args.h5ad)
    if not p.is_file():
        sys.stderr.write(f"ERROR: not found: {p}\n")
        return 2

    print(f"h5ad: {p}  ({p.stat().st_size / 1e9:.2f} GB)", flush=True)
    with h5py.File(p, "r") as f:
        print("\n=== /X layout ===", flush=True)
        dump_x_layout(f, args.n_data_preview)

        print("\n=== /obs keys ===", flush=True)
        print(f"  {list(f['obs'].keys())}", flush=True)

        for col in OBS_COLS:
            print(f"\n=== obs/{col} ===", flush=True)
            dump_obs_value_counts(f, col)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
