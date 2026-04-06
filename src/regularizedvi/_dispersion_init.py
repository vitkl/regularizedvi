"""Data-driven initialization of NB dispersion (theta) via method of moments.

Decomposes empirical gene variance into Poisson, library-size, and gene-specific
components. The gene-specific excess (divided by a biological variance fraction)
provides a per-gene theta estimate for initializing px_r_mu.

Uses chunked h5py + Welford's online algorithm for memory-efficient computation
on large datasets (100k–1M+ cells).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp


def _decode(v):
    """Decode bytes to str if needed."""
    return v.decode() if isinstance(v, bytes) else v


def _read_sparse_matrix(group):
    """Read a sparse matrix from an h5py group (CSR or CSC)."""
    encoding = _decode(group.attrs.get("encoding-type", b""))
    data = group["data"]
    indices = group["indices"]
    indptr = group["indptr"]
    shape = tuple(group.attrs["shape"])

    if "csr" in encoding:
        return sp.csr_matrix((data[:], indices[:], indptr[:]), shape=shape)
    elif "csc" in encoding:
        return sp.csc_matrix((data[:], indices[:], indptr[:]), shape=shape)
    else:
        return sp.csr_matrix((data[:], indices[:], indptr[:]), shape=shape)


def _get_feature_mask(f, feature_type):
    """Get boolean mask for features matching feature_type in var/feature_types."""
    var = f["var"]
    ft = var["feature_types"]
    if isinstance(ft, h5py.Group) and "categories" in ft:
        cats = [_decode(c) for c in ft["categories"][:]]
        codes = ft["codes"][:]
        target_idx = cats.index(feature_type)
        mask = codes == target_idx
    else:
        vals = np.array([_decode(v) for v in ft[:]])
        mask = vals == feature_type
    return mask


def _load_matrix(f, layer, feature_mask=None):
    """Load X or a named layer, optionally subset columns by feature_mask."""
    if layer is None or layer == "X":
        src = f["X"]
    else:
        src = f["layers"][layer]

    if isinstance(src, h5py.Group):
        mat = _read_sparse_matrix(src)
    else:
        mat = src[:]

    if feature_mask is not None:
        if sp.issparse(mat):
            mat = mat[:, feature_mask]
        else:
            mat = mat[:, feature_mask]

    return mat


def _read_obs_column(f, col_name):
    """Read a categorical or string column from obs."""
    obs = f["obs"]
    col = obs[col_name]
    if isinstance(col, h5py.Group) and "codes" in col:
        return col["codes"][:]
    else:
        return col[:]


def compute_dispersion_init(
    adata_or_path,
    layer: str | None = None,
    dispersion_key: str | None = None,
    biological_variance_fraction: float = 0.9,
    theta_min: float = 0.01,
    theta_max: float = 10.0,
    chunk_size: int = 5000,
    feature_type: str | None = None,
    label_key: str | None = None,
    min_cells_per_group: int = 50,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """Compute per-gene NB dispersion init from method of moments on raw counts.

    Variance decomposition (law of total variance):
        Var(x_g) = mu_g                              [A: Poisson]
                 + mu_g^2 * CV^2(L)                  [B+C: library size]
                 + (mu_g^2/theta_g) * (1 + CV^2(L))  [D: gene-specific NB]

    Solving for theta:
        excess_raw = Var(x_g) - mu_g - mu_g^2 * CV^2(L)
        excess_adjusted = excess_raw / (1 + CV^2(L))
        excess_technical = excess_adjusted * (1 - biological_variance_fraction)
        theta_g = mu_g^2 / excess_technical

    Parameters
    ----------
    adata_or_path
        AnnData object or path to .h5ad file.
    layer
        Layer to use. None or "X" for X matrix.
    dispersion_key
        obs column for batch/dispersion grouping. Used to compute
        within-batch vs between-batch library size variance.
    biological_variance_fraction
        Fraction of excess variance assumed biological (default 0.9 = 90% biological).
        Technical fraction = 1 - biological_variance_fraction.
        Higher values → larger theta (less overdispersion attributed to technical).
    theta_min
        Lower clamp for theta on linear scale before log transform.
    theta_max
        Upper clamp for theta on linear scale before log transform.
    chunk_size
        Number of cells per chunk for Welford computation.
    feature_type
        Filter by var['feature_types'] (e.g. 'GEX' for RNA, 'Peaks' for ATAC).
    label_key
        obs column for cell type labels. When provided, also computes per-group
        mean/variance/theta for within-cell-type analysis. Groups with NaN/empty
        labels or fewer than ``min_cells_per_group`` cells are excluded.
    min_cells_per_group
        Minimum cells required per group when ``label_key`` is set.
    verbose
        Print progress.

    Returns
    -------
    log_theta : np.ndarray
        log(clipped theta) array of shape (n_genes,), suitable for px_r_init_mean.
    diagnostics : dict
        Unclipped theta arrays and intermediate quantities for inspection.
        When ``label_key`` is set, also contains per-group arrays:
        ``mean_g_per_group``, ``var_g_per_group``, ``theta_per_group``, ``group_sizes``.
    """
    # Resolve input — only AnnData path supports label_key for now
    if isinstance(adata_or_path, (str, Path)):
        if label_key is not None:
            raise ValueError(
                "label_key is only supported for in-memory AnnData, not h5ad paths. Load the AnnData first."
            )
        path = str(adata_or_path)
        return _compute_from_h5ad(
            path,
            layer,
            dispersion_key,
            biological_variance_fraction,
            theta_min,
            theta_max,
            chunk_size,
            feature_type,
            verbose,
        )
    else:
        if feature_type is not None:
            raise ValueError(
                "feature_type filtering is only supported for h5ad file paths, "
                "not in-memory AnnData. Subset your AnnData before calling."
            )
        return _compute_from_anndata(
            adata_or_path,
            layer,
            dispersion_key,
            biological_variance_fraction,
            theta_min,
            theta_max,
            chunk_size,
            label_key,
            min_cells_per_group,
            verbose,
        )


def _compute_from_h5ad(
    path,
    layer,
    dispersion_key,
    biological_variance_fraction,
    theta_min,
    theta_max,
    chunk_size,
    feature_type,
    verbose,
):
    """Compute dispersion init from h5ad file using chunked h5py."""
    with h5py.File(path, "r") as f:
        feature_mask = None
        if feature_type is not None:
            feature_mask = _get_feature_mask(f, feature_type)

        mat = _load_matrix(f, layer, feature_mask)

        batch_codes = None
        if dispersion_key is not None:
            batch_codes = _read_obs_column(f, dispersion_key)

    n_cells, n_genes = mat.shape
    if verbose:
        print(f"Computing dispersion init: {n_cells} cells x {n_genes} genes")

    # --- Pass 1: Library sizes and batch statistics ---
    lib_sizes = np.zeros(n_cells, dtype=np.float64)
    n_chunks = (n_cells + chunk_size - 1) // chunk_size

    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_cells)
        if sp.issparse(mat):
            lib_sizes[start:end] = np.asarray(mat[start:end].sum(axis=1)).flatten()
        else:
            lib_sizes[start:end] = mat[start:end].sum(axis=1)

    # Global library stats
    mean_L = np.mean(lib_sizes)
    var_L = np.var(lib_sizes)
    cv2_L = var_L / (mean_L**2) if mean_L > 0 else 0.0

    # Per-batch library stats
    cv2_L_within = np.nan
    cv2_L_between = np.nan
    if batch_codes is not None:
        unique_batches = np.unique(batch_codes)
        batch_means = []
        within_var_sum = 0.0
        within_count = 0
        for b in unique_batches:
            mask = batch_codes == b
            libs_b = lib_sizes[mask]
            if len(libs_b) > 1:
                batch_means.append(np.mean(libs_b))
                within_var_sum += np.var(libs_b) * len(libs_b)
                within_count += len(libs_b)
        if within_count > 0:
            pooled_within_var = within_var_sum / within_count
            cv2_L_within = pooled_within_var / (mean_L**2) if mean_L > 0 else 0.0
        if len(batch_means) > 1:
            cv2_L_between = np.var(batch_means) / (mean_L**2) if mean_L > 0 else 0.0

    if verbose:
        print(f"  Library sizes: mean={mean_L:.1f}, CV²={cv2_L:.4f}")
        if batch_codes is not None:
            print(f"  CV²(L) within-batch={cv2_L_within:.4f}, between-batch={cv2_L_between:.4f}")

    # --- Pass 2: Per-gene mean and variance via batch Welford ---
    # Uses parallel/batch Welford: compute chunk mean+M2, then merge.
    # Fully vectorized — no Python loop over cells.
    mean_g = np.zeros(n_genes, dtype=np.float64)
    m2_g = np.zeros(n_genes, dtype=np.float64)
    count = 0

    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_cells)

        if sp.issparse(mat):
            chunk = mat[start:end].toarray().astype(np.float64)
        else:
            chunk = mat[start:end].astype(np.float64)

        # Batch Welford: merge chunk stats into running accumulators
        n_chunk = chunk.shape[0]
        chunk_mean = chunk.mean(axis=0)
        chunk_m2 = ((chunk - chunk_mean) ** 2).sum(axis=0)

        # Combine running (count, mean_g, m2_g) with chunk (n_chunk, chunk_mean, chunk_m2)
        new_count = count + n_chunk
        delta = chunk_mean - mean_g
        mean_g = (count * mean_g + n_chunk * chunk_mean) / new_count
        m2_g = m2_g + chunk_m2 + (delta**2) * count * n_chunk / new_count
        count = new_count

        if verbose and ((i + 1) % 20 == 0 or (i + 1) == n_chunks):
            print(f"  Welford pass: {min(end, n_cells)}/{n_cells} cells")

    var_g = m2_g / count

    # --- Compute theta ---
    return _compute_theta(
        mean_g,
        var_g,
        cv2_L,
        cv2_L_within,
        cv2_L_between,
        biological_variance_fraction,
        theta_min,
        theta_max,
        verbose,
    )


def _compute_from_anndata(
    adata,
    layer,
    dispersion_key,
    biological_variance_fraction,
    theta_min,
    theta_max,
    chunk_size,
    label_key,
    min_cells_per_group,
    verbose,
):
    """Compute dispersion init from in-memory AnnData."""
    if layer is not None and layer != "X":
        X = adata.layers[layer]
    else:
        X = adata.X

    n_cells, n_genes = X.shape
    if verbose:
        print(f"Computing dispersion init: {n_cells} cells x {n_genes} genes")

    # Library sizes
    if sp.issparse(X):
        lib_sizes = np.asarray(X.sum(axis=1)).flatten().astype(np.float64)
    else:
        lib_sizes = X.sum(axis=1).astype(np.float64)

    mean_L = np.mean(lib_sizes)
    var_L = np.var(lib_sizes)
    cv2_L = var_L / (mean_L**2) if mean_L > 0 else 0.0

    # Per-batch stats
    cv2_L_within = np.nan
    cv2_L_between = np.nan
    if dispersion_key is not None and dispersion_key in adata.obs.columns:
        batch_codes = adata.obs[dispersion_key].values
        unique_batches = np.unique(batch_codes)
        batch_means = []
        within_var_sum = 0.0
        within_count = 0
        for b in unique_batches:
            mask = batch_codes == b
            libs_b = lib_sizes[mask]
            if len(libs_b) > 1:
                batch_means.append(np.mean(libs_b))
                within_var_sum += np.var(libs_b) * len(libs_b)
                within_count += len(libs_b)
        if within_count > 0:
            pooled_within_var = within_var_sum / within_count
            cv2_L_within = pooled_within_var / (mean_L**2) if mean_L > 0 else 0.0
        if len(batch_means) > 1:
            cv2_L_between = np.var(batch_means) / (mean_L**2) if mean_L > 0 else 0.0

    if verbose:
        print(f"  Library sizes: mean={mean_L:.1f}, CV²={cv2_L:.4f}")

    # Per-gene mean and variance via batch Welford (chunked, fully vectorized)
    mean_g = np.zeros(n_genes, dtype=np.float64)
    m2_g = np.zeros(n_genes, dtype=np.float64)
    count = 0

    n_chunks = (n_cells + chunk_size - 1) // chunk_size
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_cells)

        if sp.issparse(X):
            chunk = X[start:end].toarray().astype(np.float64)
        else:
            chunk = np.asarray(X[start:end], dtype=np.float64)

        n_chunk = chunk.shape[0]
        chunk_mean = chunk.mean(axis=0)
        chunk_m2 = ((chunk - chunk_mean) ** 2).sum(axis=0)

        new_count = count + n_chunk
        delta = chunk_mean - mean_g
        mean_g = (count * mean_g + n_chunk * chunk_mean) / new_count
        m2_g = m2_g + chunk_m2 + (delta**2) * count * n_chunk / new_count
        count = new_count

    var_g = m2_g / count

    log_theta, diagnostics = _compute_theta(
        mean_g,
        var_g,
        cv2_L,
        cv2_L_within,
        cv2_L_between,
        biological_variance_fraction,
        theta_min,
        theta_max,
        verbose,
    )

    # Per-group (cell type) statistics when label_key is provided
    if label_key is not None and label_key in adata.obs.columns:
        import pandas as pd

        labels = adata.obs[label_key].values
        # Filter out NaN/None/empty labels
        valid_mask = pd.notna(labels)
        if hasattr(labels, "astype"):
            str_labels = pd.Series(labels).astype(str)
            valid_mask = valid_mask & (str_labels != "") & (str_labels != "nan") & (str_labels != "None")
            labels_clean = str_labels.values
        else:
            labels_clean = np.array([str(x) for x in labels])

        unique_groups = [g for g in np.unique(labels_clean[valid_mask]) if g not in ("", "nan", "None")]

        mean_g_per_group = {}
        var_g_per_group = {}
        theta_per_group = {}
        group_sizes = {}
        eps_theta = 1e-10

        for g_name in unique_groups:
            g_mask = (labels_clean == g_name) & valid_mask
            n_g = int(g_mask.sum())
            if n_g < min_cells_per_group:
                continue

            X_g = X[g_mask]
            if sp.issparse(X_g):
                X_g = X_g.toarray()
            X_g = np.asarray(X_g, dtype=np.float64)

            mean_gk = X_g.mean(axis=0)
            var_gk = X_g.var(axis=0)  # population variance (ddof=0)

            # Within-group excess and theta
            excess_gk = var_gk - mean_gk  # Poisson-subtracted
            theta_gk = mean_gk**2 / np.maximum(excess_gk, eps_theta)
            theta_gk = np.clip(theta_gk, theta_min, theta_max)

            mean_g_per_group[g_name] = mean_gk.astype(np.float32)
            var_g_per_group[g_name] = var_gk.astype(np.float32)
            theta_per_group[g_name] = theta_gk.astype(np.float32)
            group_sizes[g_name] = n_g

        diagnostics["mean_g_per_group"] = mean_g_per_group
        diagnostics["var_g_per_group"] = var_g_per_group
        diagnostics["theta_per_group"] = theta_per_group
        diagnostics["group_sizes"] = group_sizes

        if verbose:
            n_valid = len(group_sizes)
            n_total = len(unique_groups)
            print(f"\n  Per-group stats ({label_key}): {n_valid}/{n_total} groups with >={min_cells_per_group} cells")
            for g_name in sorted(group_sizes, key=lambda k: -group_sizes[k])[:5]:
                print(
                    f"    {g_name}: {group_sizes[g_name]} cells, theta median={np.median(theta_per_group[g_name]):.3f}"
                )
            if n_valid > 5:
                print(f"    ... and {n_valid - 5} more groups")

    return log_theta, diagnostics


def _compute_theta(
    mean_g,
    var_g,
    cv2_L,
    cv2_L_within,
    cv2_L_between,
    biological_variance_fraction,
    theta_min,
    theta_max,
    verbose,
):
    """Compute theta from per-gene mean/variance and library stats.

    Returns (log_theta_clipped, diagnostics_dict).
    """
    eps = 1e-10

    # Step A: Poisson variance
    poisson_var = mean_g

    # Step B+C: Library size variance
    library_var = (mean_g**2) * cv2_L

    # Step D: Excess (gene-specific, beyond Poisson + library)
    excess_raw = var_g - poisson_var - library_var

    # Correction: NB overdispersion inflated by library variability
    nb_inflation = 1 + cv2_L
    excess_adjusted = excess_raw / nb_inflation

    # Shrinkage: only technical fraction of excess attributed to NB dispersion
    technical_fraction = 1 - biological_variance_fraction
    excess_technical = excess_adjusted * technical_fraction

    # Theta option 1: full correction
    # Sub-Poisson genes have excess_raw < 0 → excess_technical < 0
    # After max(..., eps), these get theta → very large (near-Poisson), which is correct
    theta_option1 = (mean_g**2) / np.maximum(excess_technical, eps)

    # Option 2: simple (no library correction)
    excess_simple = var_g - poisson_var
    excess_technical_simple = excess_simple * technical_fraction
    theta_option2 = (mean_g**2) / np.maximum(excess_technical_simple, eps)

    # Clamp and log (use max(theta_min, 1e-10) to avoid log(0))
    _effective_min = max(theta_min, 1e-10) if theta_min > 0 else 1e-10
    theta_clamped = np.clip(theta_option1, _effective_min, theta_max)
    log_theta = np.log(theta_clamped).astype(np.float32)

    n_sub_poisson = int((excess_raw <= 0).sum())

    if verbose:
        q = [0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
        q_labels = ["min", "1%", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "99%", "max"]
        print(f"\n  Sub-Poisson genes: {n_sub_poisson}/{len(mean_g)}")
        print(f"  CV²(L) = {cv2_L:.4f}, 1+CV² = {nb_inflation:.4f}")
        print("\n  Theta option 1 (full correction) quantiles:")
        for ql, qv in zip(q_labels, np.quantile(theta_option1, q), strict=True):
            print(f"    {ql:>5s}: {qv:.4f}")
        print("\n  Theta option 2 (simple) quantiles:")
        for ql, qv in zip(q_labels, np.quantile(theta_option2, q), strict=True):
            print(f"    {ql:>5s}: {qv:.4f}")
        print(f"\n  log(theta) clipped [{theta_min}, {theta_max}] quantiles:")
        for ql, qv in zip(q_labels, np.quantile(log_theta, q), strict=True):
            print(f"    {ql:>5s}: {qv:.4f}")

    diagnostics = {
        "theta_option1": theta_option1,
        "theta_option2": theta_option2,
        "mean_g": mean_g,
        "var_g": var_g,
        "cv2_L": cv2_L,
        "cv2_L_within_batch": cv2_L_within,
        "cv2_L_between_batch": cv2_L_between,
        "excess_raw": excess_raw,
        "excess_adjusted": excess_adjusted,
        "poisson_var": poisson_var,
        "library_var": library_var,
        "n_sub_poisson": n_sub_poisson,
        "theta_min": theta_min,
        "theta_max": theta_max,
        "biological_variance_fraction": biological_variance_fraction,
    }

    return log_theta, diagnostics


def compute_bursting_init(
    adata_or_path,
    layer: str | None = None,
    dispersion_key: str | None = None,
    biological_variance_fraction: float = 0.9,
    burst_size_intercept: float = 1.0,
    sensitivity: float = 1.0,
    dispersion_hyper_prior_alpha: float = 2.0,
    theta_min: float = 0.01,
    theta_max: float = 10.0,
    chunk_size: int = 5000,
    feature_type: str | None = None,
    verbose: bool = True,
) -> tuple[dict, dict]:
    """Compute init values for burst_frequency_size decoder.

    Returns per-gene initial values for:
    - stochastic_v_scale: std of technical variance (count-space)
    - burst_freq bias: inv_softplus(burst_frequency) for decoder bias init
    - burst_size bias: inv_softplus(burst_size - intercept) for decoder bias init

    Uses the same MoM variance decomposition as compute_dispersion_init,
    then splits excess into biological (burst_freq) and technical (stochastic_v).

    Parameters
    ----------
    adata_or_path
        AnnData object or path to .h5ad file.
    layer
        Layer to use. None or "X" for X matrix.
    dispersion_key
        obs column for batch/dispersion grouping.
    biological_variance_fraction
        Fraction of excess variance attributed to biology (default 0.9).
    burst_size_intercept
        The intercept added to softplus(burst_size_decoder_output).
    sensitivity
        Library centering sensitivity (e.g. 1.0 for RNA, 0.2 for ATAC).
        Used to normalize burst_size to rate-space.
    dispersion_hyper_prior_alpha
        Alpha of the Gamma hyper-prior on stochastic_v rate. Used to compute
        suggested_hyper_beta from MoM estimates.
    theta_min, theta_max
        Clamp range for burst_freq (same as theta).
    chunk_size
        Cells per chunk for Welford computation.
    feature_type
        Filter by var['feature_types'] (e.g. 'GEX').
    verbose
        Print progress.

    Returns
    -------
    init_values : dict
        Keys: 'stochastic_v_scale', 'burst_freq_bias', 'burst_size_bias',
        'log_theta', 'burst_freq', 'burst_size', 'suggested_hyper_beta'.
        All arrays are shape (n_genes,) except suggested_hyper_beta (scalar).
    diagnostics : dict
        Intermediate quantities for inspection.
    """
    # First compute theta via existing MoM (reuse all the machinery)
    log_theta, diagnostics = compute_dispersion_init(
        adata_or_path,
        layer=layer,
        dispersion_key=dispersion_key,
        biological_variance_fraction=biological_variance_fraction,
        theta_min=theta_min,
        theta_max=theta_max,
        chunk_size=chunk_size,
        feature_type=feature_type,
        verbose=verbose,
    )

    mean_g = diagnostics["mean_g"]
    excess_adjusted = diagnostics["excess_adjusted"]
    eps = 1e-4

    # Split excess into biological and technical components
    technical_fraction = 1 - biological_variance_fraction
    excess_technical = np.maximum(excess_adjusted * technical_fraction, eps**2)
    excess_biological = np.maximum(excess_adjusted * biological_variance_fraction, eps**2)

    # stochastic_v: scale param (std-like), then .pow(2) gives variance in the model
    sv_scale = np.sqrt(excess_technical)

    # Auto-derive hyper-prior beta from MoM: E[lambda] = 1/median(sv_scale)
    _median_sv = float(np.median(sv_scale))
    suggested_hyper_beta = dispersion_hyper_prior_alpha * max(_median_sv, eps)

    # burst_freq: biological concentration = mean^2 / excess_biological
    burst_freq = np.clip(mean_g**2 / excess_biological, theta_min, theta_max)

    # burst_size: rate-space (divide by sensitivity), then subtract intercept, then inv_softplus
    valid = burst_freq > eps
    burst_size_total = np.where(valid, mean_g / (burst_freq * sensitivity), 0.0)
    default_burst_size = np.min(burst_size_total[valid]) if np.any(valid) else 1.0
    burst_size_raw_val = np.where(valid, burst_size_total, default_burst_size) - burst_size_intercept
    burst_size_raw_val = np.maximum(burst_size_raw_val, eps)
    burst_size_bias = np.log(np.expm1(burst_size_raw_val)).astype(np.float32)  # inv_softplus

    # burst_freq bias: inv_softplus(burst_freq) for decoder bias init
    burst_freq_clamped = np.maximum(burst_freq, eps)
    burst_freq_bias = np.log(np.expm1(burst_freq_clamped)).astype(np.float32)  # inv_softplus

    if verbose:
        print("\n  Bursting model init:")
        print(f"    stochastic_v scale: median={np.median(sv_scale):.4f}, mean={np.mean(sv_scale):.4f}")
        print(f"    burst_freq: median={np.median(burst_freq):.4f}, mean={np.mean(burst_freq):.4f}")
        print(
            f"    burst_size (rate-space, sens={sensitivity}): "
            f"median={np.median(burst_size_total[valid]):.4f}, "
            f"mean={np.mean(burst_size_total[valid]):.4f}"
        )
        print(f"    genes with valid burst_size: {np.sum(valid)}/{len(mean_g)}")
        _suggested_lambda = dispersion_hyper_prior_alpha / suggested_hyper_beta
        print(
            f"    suggested hyper-prior: Gamma({dispersion_hyper_prior_alpha}, {suggested_hyper_beta:.4f})"
            f" → E[λ]={_suggested_lambda:.1f}"
        )

    init_values = {
        "burst_freq_bias": burst_freq_bias,
        "burst_size_bias": burst_size_bias,
        "log_theta": log_theta,
        "burst_freq": burst_freq.astype(np.float32),
        "burst_size": burst_size_total.astype(np.float32),
        "stochastic_v_scale": sv_scale.astype(np.float32),
        "suggested_hyper_beta": suggested_hyper_beta,
    }

    return init_values, diagnostics
