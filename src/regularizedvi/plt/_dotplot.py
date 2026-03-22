"""Per-dataset grouped dotplots of marker genes."""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc


def plot_marker_dotplots(
    adata,
    marker_csv: str | Path,
    groupby: str,
    layer: str = "norm_counts",
    dataset_col: str = "dataset",
    symbol_col: str = "SYMBOL",
    save_dir: str | Path | None = None,
    figsize_per_group: tuple[float, float] = (0.4, 0.35),
    **kwargs,
) -> list:
    """Plot scanpy dotplots of marker genes per dataset, grouped by cell type.

    Parameters
    ----------
    adata
        AnnData with gene expression and obs columns.
    marker_csv
        Path to CSV with columns: gene, cell_type, lineage, category.
    groupby
        Obs column to group cells by (e.g. "leiden", "harmonized_annotation").
    layer
        Layer to use for expression values.
    dataset_col
        Obs column with dataset identifiers.
    symbol_col
        Var column mapping var_names to gene symbols.
    save_dir
        Directory to save figures. If None, figures are shown inline.
    figsize_per_group
        (width_per_gene, height_per_group) scaling for figure size.
    **kwargs
        Passed to ``sc.pl.dotplot``.

    Returns
    -------
    List of (dataset_name, DotPlot) tuples for successfully plotted datasets.
    """
    marker_df = pd.read_csv(marker_csv)

    # Build ordered dict: cell_type -> [gene_symbol, ...]
    gene_groups = OrderedDict()
    for _, row in marker_df.iterrows():
        ct = row["cell_type"]
        gene = row["gene"]
        if ct not in gene_groups:
            gene_groups[ct] = []
        if gene not in gene_groups[ct]:
            gene_groups[ct].append(gene)

    # Map gene symbols to var_names
    if symbol_col in adata.var.columns:
        symbol_to_var = dict(zip(adata.var[symbol_col], adata.var_names, strict=False))
    else:
        symbol_to_var = {v: v for v in adata.var_names}

    # Filter to genes present in adata
    var_names_grouped = OrderedDict()
    for ct, genes in gene_groups.items():
        present = [symbol_to_var[g] for g in genes if g in symbol_to_var]
        if present:
            var_names_grouped[ct] = present

    if not var_names_grouped:
        print("No marker genes found in adata — skipping dotplots")
        return []

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    results = []
    datasets = sorted(adata.obs[dataset_col].unique())

    for ds in datasets:
        mask = adata.obs[dataset_col] == ds
        adata_sub = adata[mask]
        n_cells = mask.sum()

        # Skip if groupby has ≤1 unique value
        n_groups = adata_sub.obs[groupby].nunique()
        if n_groups <= 1:
            print(f"  {ds}: only {n_groups} group(s) in '{groupby}', skipping")
            continue

        # Check layer exists
        if layer not in adata_sub.layers and layer is not None:
            print(f"  {ds}: layer '{layer}' not found, using X")
            _layer = None
        else:
            _layer = layer

        print(f"\n{ds} (n={n_cells}, {n_groups} groups)")
        try:
            dp = sc.pl.dotplot(
                adata_sub,
                var_names=var_names_grouped,
                groupby=groupby,
                layer=_layer,
                title=f"{ds} (n={n_cells})",
                show=False,
                return_fig=True,
                **kwargs,
            )
            dp.make_figure()

            if save_dir is not None:
                safe_name = ds.replace("/", "_").replace(" ", "_")
                fig_path = os.path.join(save_dir, f"dotplot_{groupby}_{safe_name}.png")
                dp.savefig(fig_path, dpi=150, bbox_inches="tight")
                print(f"  Saved to {fig_path}")

            plt.show()
            results.append((ds, dp))
        except Exception as e:  # noqa: BLE001
            print(f"  {ds}: dotplot failed: {e}")

    return results
