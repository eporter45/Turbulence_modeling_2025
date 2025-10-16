# -*- coding: utf-8 -*-
"""
Scratch script to test plot_triplet with loaded shear mixing data.

@author: eoporter
"""

import os
import torch
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from PreProcess.Load_and_norm import load_and_norm
from Plotting.Plot_preds import plot_triplet

# === Config ===
cfg = {
    "trial_name": "single_inter_c2_f2",   # <-- adjust if needed
    "paths": {
        "output_dir": "Pipelines/RST/outputs",
    },
    "features": {
        "dnd": "nondim",
        "grad_type": "MLS",
        "input": "FS9",
        "norm": "global",
        "output": ["uu", "uv", "vv", "uw", "vw", "ww"],
        "y_norm": False,
        "in_is_out": False,
        "trim_z": True,
    },
}

# === Load a small batch of data ===
dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data", "Shear_mixing"))
exp_dir = os.path.join(dataset_dir, 'EXP',"train_exp")
rans_dir = os.path.join(dataset_dir, "RANS", "training")

data_bundle = load_and_norm(cfg, exp_dir, rans_dir)

# Grab one test case for quick plotting
case_name, df_truth = list(data_bundle["y_test"].items())[0]
df_pred = df_truth.copy()  # Fake prediction = truth + noise
df_pred += 0.1 * torch.randn_like(torch.tensor(df_pred.values)).numpy()

# Extract grid and one tensor component (uu for test)
cx, cy = data_bundle["grid_dict"]["test"][case_name]
triang = tri.Triangulation(cx, cy)

truth = df_truth["uv"].values
pred = df_pred["uv"].values

# === Plot ===
fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
plot_triplet(axs, triang, truth, pred, title_prefix="uu", vmin=None, vmax=None)
fig.suptitle(f"Scratch test: {case_name}", fontsize=14)

plt.show()
plt.close()
