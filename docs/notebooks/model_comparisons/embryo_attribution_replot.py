"""Re-plot attribution UMAP comparison grids for all completed embryo runs.

Loads only ``obs`` and ``obsm`` from each ``adata_trained.h5ad`` (35-53 GB on disk)
via h5py + ``anndata.io.read_elem``, rebuilds a tiny in-memory AnnData with no
``X``/``var``/``layers``, and renders the same ``plot_umap_comparison`` grid the
training notebook produces — at ``figsize_per_panel=(15, 15)`` so panels are
visually comparable across runs.

PNGs land inside the per-run results dir as ``<run_name>_<color_var>.png`` and
are symlinked into ``embryo_attribution_figures/`` next to this script.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import traceback

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from anndata.io import read_elem

from regularizedvi import RegularizedMultimodalVI

RESULTS_ROOT = pathlib.Path(
    "/nemo/lab/briscoej/home/users/kleshcv/cell2state_embryo_data/results/regularizedvi_embryo_v1"
)
SCRIPT_DIR = pathlib.Path(__file__).parent
TSVS = [
    SCRIPT_DIR / "embryo_rna_only_slurm_jobs.tsv",
    SCRIPT_DIR / "embryo_slurm_jobs.tsv",
]
SHARED_DIR = SCRIPT_DIR / "embryo_attribution_figures"
COLOR_VARS = ["Experiment", "Embryo", "Section", "cell_type_lvl2_new", "dv_domain_label_v2"]
FIGSIZE_PER_PANEL = (15, 15)


def load_obs_obsm_only(h5ad_path: pathlib.Path) -> ad.AnnData:
    with h5py.File(h5ad_path, "r") as f:
        obs = read_elem(f["obs"])
        obsm = {k: read_elem(f["obsm"][k]) for k in f["obsm"].keys()}
    return ad.AnnData(obs=obs, obsm=obsm)


def build_umap_keys(adata: ad.AnnData) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    if "X_umap_joint" in adata.obsm:
        keys.append(("X_umap_joint", "Joint Z"))
    modalities = sorted(
        {
            k.removeprefix("X_umap_")
            for k in adata.obsm
            if k.startswith("X_umap_")
            and not k.startswith("X_umap_attr_")
            and k != "X_umap_joint"
        }
    )
    for name in modalities:
        for k, label in [
            (f"X_umap_{name}", f"{name} Z"),
            (f"X_umap_attr_{name}", f"{name} attribution Z"),
            (f"X_umap_attr_{name}_with_covs", f"{name} attr (with covs)"),
        ]:
            if k in adata.obsm:
                keys.append((k, label))
    return keys


def collect_run_names() -> list[str]:
    names: list[str] = []
    for tsv in TSVS:
        df = pd.read_csv(tsv, sep="\t")
        names.extend(df["name"].tolist())
    return names


def replot_run(run_name: str, color_vars: list[str]) -> None:
    h5ad = RESULTS_ROOT / run_name / "adata_trained.h5ad"
    if not h5ad.exists():
        print(f"SKIP {run_name}: no adata_trained.h5ad", flush=True)
        return
    print(f"Loading {run_name} ...", flush=True)
    adata = load_obs_obsm_only(h5ad)
    umap_keys = build_umap_keys(adata)
    print(f"  obsm umap_keys: {[k for k, _ in umap_keys]}", flush=True)
    for color_var in color_vars:
        if color_var not in adata.obs.columns:
            print(f"  SKIP {color_var}: missing in obs", flush=True)
            continue
        try:
            fig = RegularizedMultimodalVI.plot_umap_comparison(
                None,
                adata,
                color=[color_var],
                umap_keys=umap_keys,
                figsize_per_panel=FIGSIZE_PER_PANEL,
            )
        except Exception:
            print(f"  ERROR plotting {color_var} for {run_name}:", flush=True)
            traceback.print_exc()
            continue
        out = RESULTS_ROOT / run_name / f"{run_name}_{color_var}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        link = SHARED_DIR / out.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(out)
        print(f"  saved {out.name}", flush=True)
    del adata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N runs (smoke test).",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Process a single run by name (overrides --limit).",
    )
    args = parser.parse_args()

    SHARED_DIR.mkdir(exist_ok=True)

    if args.run:
        run_names = [args.run]
    else:
        run_names = collect_run_names()
        if args.limit is not None:
            run_names = run_names[: args.limit]

    print(f"{len(run_names)} run(s) to process", flush=True)
    for run_name in run_names:
        replot_run(run_name, COLOR_VARS)
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
