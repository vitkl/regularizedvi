"""Extract library_log_means/vars and key parameter stats from saved model checkpoints."""

import sys

import torch

paths = {
    "single_modal": "docs/notebooks/results/regularizedvi_gamma_poisson_early_stopping/model/model.pt",
    "multimodal": "results/multimodal_tutorial_early_stopping/model/model.pt",
}

# Allow filtering to specific model via CLI arg
if len(sys.argv) > 1:
    filter_key = sys.argv[1]
    paths = {k: v for k, v in paths.items() if filter_key in k}

for label, path in paths.items():
    pt = torch.load(path, map_location="cpu", weights_only=False)
    sd = pt["model_state_dict"]
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    # Library size buffers
    lib_keys = sorted(k for k in sd if "library_log" in k)
    for k in lib_keys:
        v = sd[k]
        print(f"\n  {k}: shape={tuple(v.shape)}")
        print(f"    log-scale values: {[f'{x:.4f}' for x in v.flatten().tolist()]}")
        print(f"    exp(values) [count scale]: {[f'{x:.1f}' for x in torch.exp(v).flatten().tolist()]}")

    # px_r_mu / px_r_log_sigma stats (variational posterior) + legacy px_r
    px_r_keys = sorted(k for k in sd if k.startswith("px_r"))
    for k in px_r_keys:
        v = sd[k].float()
        theta = torch.exp(v)
        print(f"\n  {k}: shape={tuple(v.shape)}")
        print(f"    log-scale: mean={v.mean():.4f}, std={v.std():.4f}, min={v.min():.4f}, max={v.max():.4f}")
        print(f"    theta=exp(px_r): mean={theta.mean():.2f}, min={theta.min():.4f}, max={theta.max():.2f}")

    # Additive background stats
    bg_keys = sorted(k for k in sd if "additive_background" in k)
    for k in bg_keys:
        v = sd[k].float()
        print(f"\n  {k}: shape={tuple(v.shape)}")
        print(f"    log-scale: mean={v.mean():.4f}, std={v.std():.4f}, min={v.min():.4f}, max={v.max():.4f}")
        print(
            f"    exp(bg): mean={torch.exp(v).mean():.6f}, min={torch.exp(v).min():.6f}, max={torch.exp(v).max():.4f}"
        )

    # Dispersion prior rate
    disp_keys = sorted(k for k in sd if "dispersion_prior" in k)
    for k in disp_keys:
        v = sd[k].float()
        rate = torch.nn.functional.softplus(v)
        print(f"\n  {k}: raw values={[f'{x:.4f}' for x in v.flatten().tolist()]}")
        print(f"    softplus(raw) [learned rate]: {[f'{x:.4f}' for x in rate.flatten().tolist()]}")

    # Feature scaling stats
    fs_keys = sorted(k for k in sd if "feature_scaling" in k)
    for k in fs_keys:
        v = sd[k].float()
        transformed = torch.nn.functional.softplus(v) / 0.7
        print(f"\n  {k}: shape={tuple(v.shape)}")
        print(f"    raw: mean={v.mean():.4f}, std={v.std():.4f}")
        print(
            f"    softplus/0.7: mean={transformed.mean():.4f}, std={transformed.std():.4f}, min={transformed.min():.4f}, max={transformed.max():.4f}"
        )
