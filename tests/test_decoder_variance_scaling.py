"""Tests for Feature 2: ``decoder_init_variance_target_fraction`` on ``RegularizedMultimodalVI``.

Per-feature decoder weight init is rescaled so the pre-softplus output variance
per gene equals ``fraction × clip_per_feature_variance_to_theta_bounds(
excess_biological_g, mean_g, theta_min, theta_max)``.

Bursting decoders (``decoder_type='burst_frequency_size'``) get an independent
``decoder_burst_size_init_variance_target_fraction`` for the ``burst_size_head``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import regularizedvi
from regularizedvi._dispersion_init import (
    clip_per_feature_variance_to_theta_bounds,
    compute_dispersion_init,
)

# ---- Helper builders ----


def _build(mdata, *, fraction, burst_fraction=None, dispersion_init="data", decoder_type="expected_RNA", seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
    return regularizedvi.RegularizedMultimodalVI(
        mdata,
        n_hidden=32,
        n_latent=8,
        dispersion_init=dispersion_init,
        decoder_type=decoder_type,
        decoder_init_variance_target_fraction=fraction,
        decoder_burst_size_init_variance_target_fraction=burst_fraction,
    )


# ---- Tests ----


def test_off_byte_identical_when_fraction_is_None(mdata):
    """``decoder_init_variance_target_fraction=None`` (default) leaves all
    decoder weights byte-identical to a freshly-built seeded baseline. Required
    for in-flight runs to remain reproducible."""
    m_a = _build(mdata, fraction=None, seed=42)
    m_b = _build(mdata, fraction=None, seed=42)
    sd_a = m_a.module.state_dict()
    sd_b = m_b.module.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for k in sd_a:
        torch.testing.assert_close(sd_a[k], sd_b[k], rtol=0, atol=0)


def test_target_fraction_lands_at_predicted_variance_non_burst(mdata):
    """With ``fraction=0.2`` and the default Kaiming pre-softplus variance of
    1/3, ``Var(W_row · h)`` per gene should be ``0.2 · clip(excess_biological_g)``
    where the clip floors small-mean genes at ``mean²/theta_max``.

    We compare the OFF and ON tensors directly: the Frobenius norm ratio of
    ``W_row_g_on / W_row_g_off`` should equal ``sqrt(3 · 0.2 · clipped_var_bio_g)``
    per row (since both builds use the same seed → same starting Kaiming draw).
    """
    fraction = 0.2

    m_off = _build(mdata, fraction=None, seed=11)
    m_on = _build(mdata, fraction=fraction, seed=11)

    rna_adata = mdata.mod["rna"]
    atac_adata = mdata.mod["atac"]
    for name, mod_adata in (("rna", rna_adata), ("atac", atac_adata)):
        _, diag = compute_dispersion_init(
            mod_adata,
            biological_variance_fraction=0.9,  # default
            verbose=False,
        )
        var_bio_g = clip_per_feature_variance_to_theta_bounds(
            np.asarray(diag["excess_biological"]),
            np.asarray(diag["mean_g"]),
            theta_min=0.01,
            theta_max=20.0,
        )
        expected_scale_g = np.sqrt(fraction * var_bio_g / (1.0 / 3.0))

        w_off = m_off.module.decoders[name].px_scale_decoder[0].weight.detach().cpu().numpy()
        w_on = m_on.module.decoders[name].px_scale_decoder[0].weight.detach().cpu().numpy()
        # Per-row ratio (avoid divide-by-zero by comparing norms)
        empirical_scale_g = np.linalg.norm(w_on, axis=1) / np.linalg.norm(w_off, axis=1)
        np.testing.assert_allclose(empirical_scale_g, expected_scale_g, rtol=1e-5, atol=1e-6)


def test_clipping_uses_shared_helper(mdata):
    """The decoder rescale must use ``clip_per_feature_variance_to_theta_bounds``
    so per-gene variance is bounded by ``[mean²/theta_max, mean²/theta_min]``
    (no gene escapes the theta-bound interval, regardless of how extreme its
    raw biological excess is)."""
    fraction = 0.5

    m_off = _build(mdata, fraction=None, seed=23)
    m_on = _build(mdata, fraction=fraction, seed=23)

    for name in ("rna", "atac"):
        _, diag = compute_dispersion_init(
            mdata.mod[name],
            biological_variance_fraction=0.9,
            verbose=False,
        )
        # Apply the helper directly (mirror what Feature 2 does internally).
        var_bio_g = clip_per_feature_variance_to_theta_bounds(
            np.asarray(diag["excess_biological"]),
            np.asarray(diag["mean_g"]),
            theta_min=0.01,
            theta_max=20.0,
        )
        # Theta bounds: var_min = mean²/theta_max, var_max = mean²/theta_min
        mean_g = np.asarray(diag["mean_g"])
        var_min_g = mean_g**2 / 20.0
        var_max_g = mean_g**2 / 0.01
        assert (var_bio_g >= var_min_g - 1e-10).all(), f"{name}: clipped variance falls below per-gene lower bound"
        assert (var_bio_g <= var_max_g + 1e-10).all(), f"{name}: clipped variance exceeds per-gene upper bound"

        # And the resulting scale matches sqrt(3·f·var_bio_g) per row.
        expected_scale_g = np.sqrt(fraction * var_bio_g / (1.0 / 3.0))
        w_off = m_off.module.decoders[name].px_scale_decoder[0].weight.detach().cpu().numpy()
        w_on = m_on.module.decoders[name].px_scale_decoder[0].weight.detach().cpu().numpy()
        empirical_scale_g = np.linalg.norm(w_on, axis=1) / np.linalg.norm(w_off, axis=1)
        np.testing.assert_allclose(empirical_scale_g, expected_scale_g, rtol=1e-5, atol=1e-6)


def test_bursting_two_heads_independent_fractions(mdata):
    """Bursting RNA decoder: ``px_scale_decoder[0]`` (= burst_freq) gets the
    main fraction; ``burst_size_head[0]`` gets the burst_size fraction; the
    two heads scale independently."""
    main_f = 0.2
    burst_f = 0.02

    m_off = _build(
        mdata,
        fraction=None,
        burst_fraction=None,
        dispersion_init="variance_burst_size",
        decoder_type={"rna": "burst_frequency_size", "atac": "expected_RNA"},
        seed=51,
    )
    m_on = _build(
        mdata,
        fraction=main_f,
        burst_fraction=burst_f,
        dispersion_init="variance_burst_size",
        decoder_type={"rna": "burst_frequency_size", "atac": "expected_RNA"},
        seed=51,
    )

    # RNA modality has bursting; check both heads scale independently
    name = "rna"
    _, diag = compute_dispersion_init(
        mdata.mod[name],
        biological_variance_fraction=0.9,
        verbose=False,
    )
    var_bio_g = clip_per_feature_variance_to_theta_bounds(
        np.asarray(diag["excess_biological"]),
        np.asarray(diag["mean_g"]),
        theta_min=0.01,
        theta_max=20.0,
    )

    for head_name, fraction in (("px_scale_decoder", main_f), ("burst_size_head", burst_f)):
        head_off = getattr(m_off.module.decoders[name], head_name)[0]
        head_on = getattr(m_on.module.decoders[name], head_name)[0]
        w_off = head_off.weight.detach().cpu().numpy()
        w_on = head_on.weight.detach().cpu().numpy()
        expected_scale_g = np.sqrt(fraction * var_bio_g / (1.0 / 3.0))
        empirical_scale_g = np.linalg.norm(w_on, axis=1) / np.linalg.norm(w_off, axis=1)
        np.testing.assert_allclose(
            empirical_scale_g,
            expected_scale_g,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"{head_name} did not scale by its fraction {fraction}",
        )


def test_burst_size_fraction_None_leaves_head_default(mdata):
    """When only the main fraction is set on a bursting decoder, ``px_scale_decoder[0]``
    is rescaled but ``burst_size_head[0]`` is left at default Kaiming (i.e.
    byte-identical to the OFF baseline)."""
    main_f = 0.2

    m_off = _build(
        mdata,
        fraction=None,
        burst_fraction=None,
        dispersion_init="variance_burst_size",
        decoder_type={"rna": "burst_frequency_size", "atac": "expected_RNA"},
        seed=71,
    )
    m_on = _build(
        mdata,
        fraction=main_f,
        burst_fraction=None,  # burst_size_head left at default
        dispersion_init="variance_burst_size",
        decoder_type={"rna": "burst_frequency_size", "atac": "expected_RNA"},
        seed=71,
    )

    name = "rna"
    # px_scale_decoder rescaled
    w_off_main = m_off.module.decoders[name].px_scale_decoder[0].weight
    w_on_main = m_on.module.decoders[name].px_scale_decoder[0].weight
    assert not torch.equal(w_off_main, w_on_main), "px_scale_decoder should be rescaled when fraction is set"

    # burst_size_head untouched (byte-identical to OFF baseline)
    w_off_burst = m_off.module.decoders[name].burst_size_head[0].weight
    w_on_burst = m_on.module.decoders[name].burst_size_head[0].weight
    torch.testing.assert_close(w_off_burst, w_on_burst, rtol=0, atol=0)


def test_prior_dispersion_init_raises_with_fraction(mdata):
    """Feature 2 needs MoM diagnostics; ``dispersion_init='prior'`` does not
    compute them, so setting either fraction with ``prior`` must raise
    ``ValueError`` rather than silently fall back."""
    regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
    with pytest.raises(ValueError, match="dispersion_init"):
        regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent=4,
            dispersion_init="prior",
            decoder_init_variance_target_fraction=0.2,
        )
    with pytest.raises(ValueError, match="dispersion_init"):
        regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent=4,
            dispersion_init="prior",
            decoder_burst_size_init_variance_target_fraction=0.02,
        )
