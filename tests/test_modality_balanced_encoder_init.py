"""Tests for Feature 1: ``use_modality_balanced_encoder_init`` on ``RegularizedMultimodalVI``.

Per-modality ``mean_encoder`` / ``var_encoder`` weights are rebalanced so the
per-element weight variance equals ``1/(3·total_n_hidden)`` across all modalities
(treating the multimodal encoder as one virtual Linear of fan_in =
``total_n_hidden = sum_m n_hidden_m``).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import regularizedvi


@pytest.fixture
def mdata_unequal_n_hidden_seeded(mdata):
    """Reuse the conftest mdata fixture (RNA + ATAC) — Feature 1's effect is
    purely about the rescale factor, not data content."""
    return mdata


def _build_model(mdata, *, balance: bool, seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
    return regularizedvi.RegularizedMultimodalVI(
        mdata,
        n_hidden={"rna": 64, "atac": 32},
        n_latent={"rna": 16, "atac": 8},
        use_modality_balanced_encoder_init=balance,
    )


def test_balance_off_byte_identical_to_today(mdata_unequal_n_hidden_seeded):
    """With ``use_modality_balanced_encoder_init=False`` the module is
    byte-identical to a freshly-built module with the same seed (default
    behaviour preserved for in-flight runs)."""
    m1 = _build_model(mdata_unequal_n_hidden_seeded, balance=False, seed=42)
    m2 = _build_model(mdata_unequal_n_hidden_seeded, balance=False, seed=42)
    sd1 = m1.module.state_dict()
    sd2 = m2.module.state_dict()
    assert sd1.keys() == sd2.keys()
    for k in sd1:
        torch.testing.assert_close(sd1[k], sd2[k], rtol=0, atol=0)


def test_balance_changes_only_mean_and_var_encoder_weights(mdata_unequal_n_hidden_seeded):
    """With balance ON, the only weights that differ from the OFF baseline are
    per-modality ``mean_encoder.weight`` and ``var_encoder.weight`` (other
    parameters are unchanged)."""
    m_off = _build_model(mdata_unequal_n_hidden_seeded, balance=False, seed=7)
    m_on = _build_model(mdata_unequal_n_hidden_seeded, balance=True, seed=7)
    sd_off = m_off.module.state_dict()
    sd_on = m_on.module.state_dict()
    assert sd_off.keys() == sd_on.keys()
    for k, v_off in sd_off.items():
        v_on = sd_on[k]
        is_z_loc_or_scale_head = k.startswith("encoders.") and (
            ".mean_encoder.weight" in k or ".var_encoder.weight" in k
        )
        if is_z_loc_or_scale_head:
            assert not torch.equal(v_off, v_on), f"{k} should be rescaled by Feature 1 but matches baseline"
        else:
            torch.testing.assert_close(v_off, v_on, rtol=0, atol=0, msg=f"{k} unexpectedly changed")


def test_per_element_weight_variance_constant_across_modalities(mdata_unequal_n_hidden_seeded):
    """After rebalancing, ``Var(W_per_element)`` of every modality's
    ``mean_encoder.weight`` equals ``1/(3·total_n_hidden)`` to within statistical
    tolerance — exactly what Kaiming would produce for a single combined Linear
    of fan_in ``total_n_hidden``.

    Statistical tolerance: each weight tensor has ``n_latent_m × n_hidden_m``
    elements drawn from a uniform distribution, so the empirical variance of
    a sample of N draws from Uniform(-a, a) (true variance ``a²/3``) has
    standard error ≈ ``a²/3 · sqrt(2/(N-1))``. We use ±25% tolerance which is
    comfortably above that for our smallest tensor (8 × 32 = 256 elements
    → ~9% SE).
    """
    n_hidden_dict = {"rna": 64, "atac": 32}
    total_n_hidden = sum(n_hidden_dict.values())
    expected_var = 1.0 / (3.0 * total_n_hidden)

    m = _build_model(mdata_unequal_n_hidden_seeded, balance=True, seed=123)
    for name in ("rna", "atac"):
        w = m.module.encoders[name].mean_encoder.weight.detach().cpu().numpy()
        emp_var = float(np.var(w))
        assert emp_var == pytest.approx(expected_var, rel=0.25), (
            f"modality {name!r}: empirical Var(W)={emp_var:.6e}, expected {expected_var:.6e} = 1/(3·{total_n_hidden})"
        )


def test_multiplier_matches_sqrt_n_hidden_ratio(mdata_unequal_n_hidden_seeded):
    """Direct check of the closed-form rescale multiplier:
    ``W_balanced = sqrt(n_hidden_m / total_n_hidden) · W_default``.

    Compares the OFF and ON tensors element-wise after dividing by the expected
    multiplier — the ratio should be exactly 1.0 (no stochastic error since both
    builds use the same seed).
    """
    n_hidden_dict = {"rna": 64, "atac": 32}
    total_n_hidden = sum(n_hidden_dict.values())

    m_off = _build_model(mdata_unequal_n_hidden_seeded, balance=False, seed=99)
    m_on = _build_model(mdata_unequal_n_hidden_seeded, balance=True, seed=99)
    for name in ("rna", "atac"):
        n_h = n_hidden_dict[name]
        expected_mult = (n_h / total_n_hidden) ** 0.5
        for head in ("mean_encoder", "var_encoder"):
            w_off = getattr(m_off.module.encoders[name], head).weight
            w_on = getattr(m_on.module.encoders[name], head).weight
            torch.testing.assert_close(w_on, expected_mult * w_off, rtol=1e-6, atol=1e-7)


def test_skip_for_single_encoder_mode(mdata_unequal_n_hidden_seeded):
    """``latent_mode='single_encoder'`` has one joint encoder (no per-modality
    encoders to rebalance); Feature 1 must be a no-op there. The state_dict is
    therefore byte-identical to the balance-OFF baseline."""
    regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata_unequal_n_hidden_seeded, batch_key="batch")
    torch.manual_seed(31)
    np.random.seed(31)
    m_off = regularizedvi.RegularizedMultimodalVI(
        mdata_unequal_n_hidden_seeded,
        n_hidden=16,
        n_latent=4,
        latent_mode="single_encoder",
        use_modality_balanced_encoder_init=False,
    )
    torch.manual_seed(31)
    np.random.seed(31)
    m_on = regularizedvi.RegularizedMultimodalVI(
        mdata_unequal_n_hidden_seeded,
        n_hidden=16,
        n_latent=4,
        latent_mode="single_encoder",
        use_modality_balanced_encoder_init=True,
    )
    sd_off = m_off.module.state_dict()
    sd_on = m_on.module.state_dict()
    assert sd_off.keys() == sd_on.keys()
    for k in sd_off:
        torch.testing.assert_close(sd_off[k], sd_on[k], rtol=0, atol=0, msg=f"{k} changed in single_encoder mode")
