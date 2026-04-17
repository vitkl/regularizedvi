"""Plotting and evaluation utilities for regularizedvi."""

from regularizedvi.plt._dotplot import plot_marker_dotplots
from regularizedvi.plt._integration_metrics import compute_integration_metrics, plot_integration_heatmap
from regularizedvi.plt._neighbourhood_correlation import (
    plot_distribution_overlap,
    plot_failure_mode_scatter,
    plot_isolation_bars,
    plot_leaf_distribution,
    plot_marker_correlation_umap,
    plot_metric_hist2d,
    plot_per_library_distributions,
)

__all__ = [
    "compute_integration_metrics",
    "plot_distribution_overlap",
    "plot_failure_mode_scatter",
    "plot_integration_heatmap",
    "plot_isolation_bars",
    "plot_leaf_distribution",
    "plot_marker_correlation_umap",
    "plot_marker_dotplots",
    "plot_metric_hist2d",
    "plot_per_library_distributions",
]
