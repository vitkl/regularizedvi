#!/usr/bin/env python3
"""Quick test: load model history and run plot_training_diagnostics."""

import matplotlib
import torch

matplotlib.use("Agg")

model_path = "results/multimodal_tutorial_early_stopping/model"
print(f"Loading model.pt from {model_path}...")
state = torch.load(f"{model_path}/model.pt", map_location="cpu", weights_only=False)

# Check what's in the saved state
print(f"Keys in model.pt: {list(state.keys())}")
if "attr_dict" in state:
    attr = state["attr_dict"]
    if "history_" in attr:
        history = attr["history_"]
        print(f"\nHistory keys ({len(history)}):")
        for k, v in history.items():
            print(f"  {k}: shape={v.shape}, dtype={v.dtypes.to_dict()}")
    else:
        print("No history_ in attr_dict")
        print(f"attr_dict keys: {list(attr.keys())}")
else:
    print(f"Top-level keys: {list(state.keys())}")
