"""Plotting helpers for ``z_dim_selection_diagnostics.ipynb``.

All helpers are pure: they take already-computed numpy / pandas data and
return a ``matplotlib.figure.Figure`` (plus ancillary summary data where
indicated). The notebook does the heavy lifting (loading models, running
``get_per_feature_reconstruction_loss`` / ``get_per_dim_kl``, etc.); this
module only renders figures and writes them to disk.

Helpers cover diagnostics D1 (per-feature NLL), D2 (posterior std at init),
and D4 (horseshoe forensics) as specified in
``~/.claude/plans/kind-wandering-bonbon.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Typing aliases used throughout.
NllDict = dict[str, pd.Series]  # {modality_name: per-feature NLL}
KlDict = dict[str, np.ndarray]  # {modality_name: per-dim KL (n_latent,)}
VariantNllDict = dict[str, NllDict]  # {variant_name: NllDict}
VariantKlDict = dict[str, KlDict]  # {variant_name: KlDict}
DatasetNllDict = dict[str, NllDict]  # {dataset_name: NllDict}


# ---------------------------------------------------------------------------
# File output helper
# ---------------------------------------------------------------------------


def save_fig(
    fig: plt.Figure,
    output_dir: str | Path | None,
    stem: str,
    suffix: str = "",
) -> None:
    """Save ``fig`` as PNG (300 DPI) + PDF under ``output_dir``.

    ``suffix`` is appended to the filename stem, e.g. ``"_phase5a"`` to
    distinguish Phase 5a previews from Phase 5b final runs.
    """
    if output_dir is None:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f"{stem}{suffix}"
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _intersect_geneset(series: pd.Series, genes: list[str] | None) -> pd.Series:
    """Return the subset of ``series`` indexed by ``genes`` (ignoring missing)."""
    if genes is None:
        return series
    present = [g for g in genes if g in series.index]
    return series.loc[present]


def _hist_with_stats(
    ax: plt.Axes,
    values: np.ndarray,
    label: str | None = None,
    color: str | None = None,
    bins: int = 60,
    alpha: float = 0.7,
) -> None:
    """Histogram with median (solid) and mean (dashed) vertical bars."""
    if len(values) == 0:
        ax.text(0.5, 0.5, "empty", transform=ax.transAxes, ha="center", va="center")
        return
    ax.hist(values, bins=bins, color=color, alpha=alpha, label=label)
    med = float(np.median(values))
    mean = float(np.mean(values))
    ax.axvline(med, color=color or "k", linestyle="-", linewidth=1.2)
    ax.axvline(mean, color=color or "k", linestyle="--", linewidth=1.2)


def _overlay_kl_bars(
    ax: plt.Axes,
    kl_per_dim: np.ndarray,
    color: str = "red",
) -> None:
    """Vertical lines at min / mean / max per-dim KL."""
    if kl_per_dim is None or len(kl_per_dim) == 0:
        return
    ax.axvline(float(np.min(kl_per_dim)), color=color, linestyle=":", linewidth=1.0)
    ax.axvline(float(np.mean(kl_per_dim)), color=color, linestyle="-", linewidth=1.0)
    ax.axvline(float(np.max(kl_per_dim)), color=color, linestyle=":", linewidth=1.0)


# ---------------------------------------------------------------------------
# D1 — per-feature NLL multi-panel histogram
# ---------------------------------------------------------------------------


def plot_per_feature_nll_panel(
    nll_per_variant: VariantNllDict,
    kl_per_dim_per_variant: VariantKlDict | None = None,
    gene_sets: dict[str, list[str] | None] | None = None,
    modality_names: list[str] | None = None,
    output_dir: str | Path | None = None,
    stem: str = "d1_per_feature_nll_panel",
    suffix: str = "",
) -> plt.Figure:
    """Multi-panel histogram: rows = variants, columns = modality × geneset.

    Each panel overlays per-dim KL bars (red) so hypothesis 1 (per-gene NLL
    vs per-dim KL cost) can be read directly.
    """
    if gene_sets is None:
        gene_sets = {"all": None}
    if modality_names is None:
        modality_names = sorted({m for v in nll_per_variant.values() for m in v})
    variants = list(nll_per_variant.keys())

    n_rows = len(variants)
    n_cols = len(modality_names) * len(gene_sets)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.4 * n_rows), squeeze=False)
    geneset_labels = list(gene_sets.keys())

    for i, variant in enumerate(variants):
        for j, modality in enumerate(modality_names):
            for k, gs_name in enumerate(geneset_labels):
                col = j * len(geneset_labels) + k
                ax = axes[i, col]
                series = nll_per_variant.get(variant, {}).get(modality)
                if series is None:
                    ax.set_visible(False)
                    continue
                subset = _intersect_geneset(series, gene_sets[gs_name])
                _hist_with_stats(ax, subset.values, color="C0")
                if kl_per_dim_per_variant is not None:
                    kl = kl_per_dim_per_variant.get(variant, {}).get(modality)
                    if kl is not None:
                        _overlay_kl_bars(ax, kl)
                if i == 0:
                    ax.set_title(f"{modality} / {gs_name}", fontsize=9)
                if col == 0:
                    ax.set_ylabel(variant, fontsize=8)
                ax.tick_params(labelsize=7)

    fig.suptitle("Per-feature NLL (rows=variant, cols=modality×geneset)", fontsize=11)
    fig.tight_layout()
    save_fig(fig, output_dir, stem, suffix)
    return fig


def plot_per_gene_ratio_histogram(
    nll_per_variant: VariantNllDict,
    kl_per_dim_per_variant: VariantKlDict,
    modality_names: list[str] | None = None,
    gene_set: list[str] | None = None,
    gene_set_name: str = "all",
    output_dir: str | Path | None = None,
    stem: str = "d1_per_gene_ratio_hist",
    suffix: str = "",
) -> tuple[plt.Figure, pd.DataFrame]:
    """Histogram of ``per_feature_nll[g] / mean_kl_per_dim`` per variant × modality.

    Reports fraction of features with ratio > 1 as a summary table.
    Returns ``(fig, summary_df)`` where ``summary_df`` has columns
    ``[variant, modality, n_features, median_ratio, frac_above_1]``.
    """
    if modality_names is None:
        modality_names = sorted({m for v in nll_per_variant.values() for m in v})
    variants = list(nll_per_variant.keys())

    rows = []
    n_rows = len(variants)
    n_cols = len(modality_names)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.2 * n_rows), squeeze=False)
    for i, variant in enumerate(variants):
        for j, modality in enumerate(modality_names):
            ax = axes[i, j]
            series = nll_per_variant.get(variant, {}).get(modality)
            kl = kl_per_dim_per_variant.get(variant, {}).get(modality)
            if series is None or kl is None or len(kl) == 0:
                ax.set_visible(False)
                continue
            subset = _intersect_geneset(series, gene_set)
            mean_kl = float(np.mean(kl))
            if mean_kl <= 0 or not np.isfinite(mean_kl):
                ax.set_visible(False)
                continue
            ratio = subset.values / mean_kl
            _hist_with_stats(ax, ratio, color="C1")
            ax.axvline(1.0, color="k", linestyle="-", linewidth=1.0)
            n_feat = int(len(ratio))
            frac_above_1 = float((ratio > 1.0).mean()) if n_feat > 0 else float("nan")
            median_ratio = float(np.median(ratio)) if n_feat > 0 else float("nan")
            rows.append(
                {
                    "variant": variant,
                    "modality": modality,
                    "n_features": n_feat,
                    "median_ratio": median_ratio,
                    "frac_above_1": frac_above_1,
                }
            )
            if i == 0:
                ax.set_title(f"{modality} / {gene_set_name}", fontsize=9)
            if j == 0:
                ax.set_ylabel(variant, fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle(f"Per-gene NLL / mean per-dim KL (gene set: {gene_set_name})", fontsize=11)
    fig.tight_layout()
    save_fig(fig, output_dir, stem, suffix)
    return fig, pd.DataFrame(rows)


def plot_cross_dataset_nll(
    nll_per_dataset: DatasetNllDict,
    modality_name: str,
    gene_set: list[str] | None = None,
    gene_set_name: str = "shared",
    output_dir: str | Path | None = None,
    stem: str = "d1_cross_dataset_nll",
    suffix: str = "",
) -> plt.Figure:
    """Overlaid per-feature NLL histograms across BM / immune / embryo.

    Only looks at one modality (typically ``"rna"``) and one gene set so the
    same gene can be tracked across datasets at different dilution factors.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"bm": "C0", "immune": "C1", "embryo": "C2"}
    for dataset_name, nll_dict in nll_per_dataset.items():
        series = nll_dict.get(modality_name)
        if series is None:
            continue
        subset = _intersect_geneset(series, gene_set)
        _hist_with_stats(
            ax,
            subset.values,
            label=f"{dataset_name} (n={len(subset)})",
            color=colors.get(dataset_name),
            alpha=0.45,
        )
    ax.set_xlabel(f"per-feature NLL ({modality_name}, {gene_set_name})")
    ax.set_ylabel("count")
    ax.set_title("Cross-dataset NLL (hypothesis 2: dilution effect)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, output_dir, stem, suffix)
    return fig


def plot_post_training_improvement(
    nll_init_per_variant: VariantNllDict,
    nll_trained_per_variant: VariantNllDict,
    mean_kl_per_dim_per_variant: dict[str, dict[str, float]],
    modality_names: list[str] | None = None,
    gene_set: list[str] | None = None,
    gene_set_name: str = "all",
    output_dir: str | Path | None = None,
    stem: str = "d1_post_training_improvement",
    suffix: str = "",
) -> tuple[plt.Figure, pd.DataFrame]:
    """Delta-NLL = (init - trained) per feature, with a per-dim KL reference line.

    Returns ``(fig, summary_df)`` where ``summary_df`` has columns
    ``[variant, modality, n_features, median_delta, mean_delta,
    mean_kl_per_dim, frac_above_kl_bar]``.
    """
    if modality_names is None:
        modality_names = sorted({m for v in nll_trained_per_variant.values() for m in v})
    variants = list(nll_trained_per_variant.keys())

    rows = []
    n_rows = len(variants)
    n_cols = len(modality_names)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.2 * n_rows), squeeze=False)
    for i, variant in enumerate(variants):
        for j, modality in enumerate(modality_names):
            ax = axes[i, j]
            trained = nll_trained_per_variant.get(variant, {}).get(modality)
            init = nll_init_per_variant.get(variant, {}).get(modality)
            if trained is None or init is None:
                ax.set_visible(False)
                continue
            # Align to common feature index before subtraction.
            common = trained.index.intersection(init.index)
            if gene_set is not None:
                common = common.intersection(pd.Index(gene_set))
            if len(common) == 0:
                ax.set_visible(False)
                continue
            delta = (init.loc[common] - trained.loc[common]).values
            _hist_with_stats(ax, delta, color="C2")
            mean_kl = float(mean_kl_per_dim_per_variant.get(variant, {}).get(modality, float("nan")))
            if np.isfinite(mean_kl):
                ax.axvline(mean_kl, color="red", linestyle="-", linewidth=1.0)
            n_feat = int(len(delta))
            median_delta = float(np.median(delta))
            mean_delta = float(np.mean(delta))
            frac_above = float((delta > mean_kl).mean()) if np.isfinite(mean_kl) and n_feat > 0 else float("nan")
            rows.append(
                {
                    "variant": variant,
                    "modality": modality,
                    "n_features": n_feat,
                    "median_delta": median_delta,
                    "mean_delta": mean_delta,
                    "mean_kl_per_dim": mean_kl,
                    "frac_above_kl_bar": frac_above,
                }
            )
            if i == 0:
                ax.set_title(f"{modality} / {gene_set_name}", fontsize=9)
            if j == 0:
                ax.set_ylabel(variant, fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle("ΔNLL = NLL_init − NLL_trained  (red line = mean per-dim KL)", fontsize=11)
    fig.tight_layout()
    save_fig(fig, output_dir, stem, suffix)
    return fig, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# D2 — posterior std at init
# ---------------------------------------------------------------------------


def plot_qz_scale_init(
    qz_scale_per_var_init_scale: dict[Any, np.ndarray],
    dataset_label: str,
    output_dir: str | Path | None = None,
    stem: str = "d2_qz_scale_init",
    suffix: str = "",
) -> plt.Figure:
    """Histogram of per-cell × per-dim ``qz.scale`` at init, one series per ``var_init_scale`` value.

    ``qz_scale_per_var_init_scale[v]`` must be a 1-D numpy array holding the
    flattened ``qz.scale`` across all cells and all latent dims. Target is
    visually obvious: ``None`` settles around 1, ``0.1`` settles around 0.1.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    for v_init, scales in qz_scale_per_var_init_scale.items():
        label = f"var_init_scale={v_init}"
        _hist_with_stats(ax, scales, label=label, alpha=0.5)
    ax.set_xlabel("qz.scale (per cell × per dim)")
    ax.set_ylabel("count")
    ax.set_title(f"Initial qz.scale — {dataset_label}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, output_dir, stem + f"_{dataset_label}", suffix)
    return fig


def plot_variability_ratio(
    qz_loc: np.ndarray,
    qz_scale: np.ndarray,
    dataset_label: str,
    var_init_scale: Any,
    output_dir: str | Path | None = None,
    stem: str = "d2_variability_ratio",
    suffix: str = "",
) -> tuple[plt.Figure, dict[str, float]]:
    """Variability-ratio test: per-dim ``std(loc) / mean(scale)``.

    Tests the user's claim that at init the cell-to-cell variability of
    ``qz.loc`` is ~1000× smaller than the magnitude of ``qz.scale``.

    Parameters
    ----------
    qz_loc, qz_scale
        Shape ``(n_cells, n_dims)``.
    dataset_label
        Used for the figure title and saved filename.
    var_init_scale
        Used for the figure title and saved filename.

    Returns
    -------
    (fig, summary)
        ``summary`` has keys ``median_ratio``, ``min_ratio``, ``max_ratio``,
        ``n_dims`` and ``claim_supported`` (True if median < 1e-2).
    """
    assert qz_loc.shape == qz_scale.shape, "qz_loc and qz_scale must have same shape"
    std_loc = np.std(qz_loc, axis=0)  # (n_dims,)
    mean_scale = np.mean(qz_scale, axis=0)  # (n_dims,)
    ratio = std_loc / np.clip(mean_scale, 1e-12, None)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(std_loc, bins=40, color="C0", alpha=0.7, label="std(qz.loc) per dim")
    axes[0].hist(mean_scale, bins=40, color="C1", alpha=0.7, label="mean(qz.scale) per dim")
    axes[0].set_xlabel("value")
    axes[0].set_ylabel("count (dims)")
    axes[0].set_title(f"{dataset_label} (var_init_scale={var_init_scale})")
    axes[0].legend(fontsize=8)

    axes[1].hist(ratio, bins=40, color="C3", alpha=0.7)
    med = float(np.median(ratio))
    axes[1].axvline(med, color="k", linestyle="-", linewidth=1.0)
    axes[1].axvline(1.0, color="red", linestyle=":", linewidth=1.0, label="ratio=1")
    axes[1].axvline(1e-3, color="gray", linestyle=":", linewidth=1.0, label="1000× claim")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("std(loc) / mean(scale) per dim (log scale)")
    axes[1].set_ylabel("count (dims)")
    axes[1].set_title(f"variability ratio (median={med:.3g})")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    save_fig(
        fig,
        output_dir,
        stem + f"_{dataset_label}_vis{var_init_scale}",
        suffix,
    )

    summary = {
        "median_ratio": med,
        "min_ratio": float(np.min(ratio)),
        "max_ratio": float(np.max(ratio)),
        "n_dims": int(len(ratio)),
        "claim_supported": bool(med < 1e-2),
    }
    return fig, summary


def plot_active_dim_counts(
    active_dim_per_model: dict[str, dict[str, float]],
    output_dir: str | Path | None = None,
    stem: str = "d2_active_dim_counts",
    suffix: str = "",
) -> plt.Figure:
    """Bar chart of final ``n_active_dims_*`` per model × modality.

    Parameters
    ----------
    active_dim_per_model
        ``{variant_name: {modality: n_active_dims}}`` — read from
        ``model.history_[f"n_active_dims_{modality}_train"].iloc[-1]``.
    """
    modality_names = sorted({m for v in active_dim_per_model.values() for m in v})
    variants = list(active_dim_per_model.keys())
    n_mods = len(modality_names)
    width = max(0.8 / n_mods, 0.2)
    x = np.arange(len(variants))

    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(variants)), 4))
    for i, modality in enumerate(modality_names):
        vals = [active_dim_per_model[v].get(modality, np.nan) for v in variants]
        ax.bar(x + i * width, vals, width=width, label=modality)
    ax.set_xticks(x + width * (n_mods - 1) / 2)
    ax.set_xticklabels(variants, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("n_active_dims (KL > 0.01)")
    ax.set_title("Final active-dim count per model × modality")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, output_dir, stem, suffix)
    return fig


# ---------------------------------------------------------------------------
# D4 — horseshoe forensics
# ---------------------------------------------------------------------------


def plot_horseshoe_init_paired(
    init_data: dict[str, dict[str, np.ndarray]],
    dataset_label: str,
    output_dir: str | Path | None = None,
    stem: str = "d4_horseshoe_init_paired",
    suffix: str = "",
) -> plt.Figure:
    """Paired histograms of initial posterior loc / scale across normal + horseshoe-old + horseshoe-fixed, on shared axes.

    ``init_data`` structure::

        {
            "normal_vis_none": {"qz_loc": array, "qz_scale": array},
            "normal_vis_0.1": {"qz_loc": array, "qz_scale": array},
            "horseshoe_old": {"qlam_loc": array, "qlam_scale": array},
            "horseshoe_fixed": {"qlam_loc": array, "qlam_scale": array},
        }

    Missing keys are simply skipped.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = {
        "normal_vis_none": "C0",
        "normal_vis_0.1": "C1",
        "horseshoe_old": "C3",
        "horseshoe_fixed": "C2",
    }
    for label, data in init_data.items():
        color = colors.get(label, None)
        # loc panel
        loc = data.get("qz_loc", data.get("qlam_loc"))
        if loc is not None:
            _hist_with_stats(axes[0], np.asarray(loc).ravel(), label=label, color=color, alpha=0.45)
        # scale panel
        scale = data.get("qz_scale", data.get("qlam_scale"))
        if scale is not None:
            _hist_with_stats(axes[1], np.asarray(scale).ravel(), label=label, color=color, alpha=0.45)
    axes[0].set_xlabel("posterior loc (per cell × per dim)")
    axes[1].set_xlabel("posterior scale (per cell × per dim)")
    axes[1].set_xscale("log")
    axes[0].set_title(f"Initial loc — {dataset_label}")
    axes[1].set_title(f"Initial scale — {dataset_label}")
    for a in axes:
        a.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, output_dir, stem + f"_{dataset_label}", suffix)
    return fig


def plot_horseshoe_zgamma_per_dim(
    zgamma_per_model: dict[str, np.ndarray],
    dataset_label: str,
    output_dir: str | Path | None = None,
    stem: str = "d4_horseshoe_zgamma_per_dim",
    suffix: str = "",
) -> plt.Figure:
    """Per-dim mean of ``z_gamma`` posterior for each horseshoe model."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, vals in zgamma_per_model.items():
        arr = np.asarray(vals).ravel()
        ax.plot(np.arange(len(arr)), np.sort(arr)[::-1], marker="o", label=label)
    ax.set_xlabel("latent dim (sorted by mean z_gamma, descending)")
    ax.set_ylabel("mean z_gamma across cells")
    ax.set_title(f"Horseshoe z_gamma per dim — {dataset_label}")
    ax.axhline(0.1, color="red", linestyle=":", label="active threshold (0.1)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, output_dir, stem + f"_{dataset_label}", suffix)
    return fig
