# -*- coding: utf-8 -*-
"""
Updated Tensor Field Visualization (for TBNN outputs)
----------------------------------------------------
Uses fast grid interpolation and pcolormesh instead of tricontourf.

Each component of a_ij, b_ij, or r_ij is plotted as:
  - Truth
  - Prediction
  - Absolute Difference

Supports:
  - Global (uniform) or per-row (component-wise) color scaling
  - Auto-interpolation to a regular grid for smooth visualization
  - Saves organized by case and tensor key
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from pathlib import Path
import torch

# --------------------------------------------------------------------------
def get_tensor_component_names(key, config):
    """Return list of tensor component names based on key and config."""
    if key == 'k':
        return ['k']
    features = config['features']['output']
    component_map = {
        'a_xx': 'xx', 'a_xy': 'xy', 'a_yy': 'yy',
        'a_xz': 'xz', 'a_yz': 'yz', 'a_zz': 'zz',
        'b_xx': 'xx', 'b_xy': 'xy', 'b_yy': 'yy',
        'b_xz': 'xz', 'b_yz': 'yz', 'b_zz': 'zz',
        'uu': 'xx', 'uv': 'xy', 'vv': 'yy',
        'uw': 'xz', 'vw': 'yz', 'ww': 'zz'
    }
    return [component_map.get(f.lower(), f) for f in features]


# --------------------------------------------------------------------------
def plot_triplet_pcolormesh(axs, cx, cy, truth, pred, title_prefix,
                            vmin=None, vmax=None, cmap='RdBu_r', grid_res=200):
    """Plot truth/pred/diff fields on common grid using pcolormesh."""
    diff = np.abs(truth - pred)

    # Make a regular mesh grid for smoother visuals
    x = np.linspace(cx.min(), cx.max(), grid_res)
    y = np.linspace(cy.min(), cy.max(), grid_res)
    X, Y = np.meshgrid(x, y)

    # Interpolate scattered data onto grid
    Z_truth = griddata((cx, cy), truth, (X, Y), method='linear')
    Z_pred  = griddata((cx, cy), pred,  (X, Y), method='linear')
    Z_diff  = np.abs(Z_truth - Z_pred)

    fields = [Z_truth, Z_pred, Z_diff]
    titles = [f"{title_prefix} Truth", f"{title_prefix} Pred", f"{title_prefix} |Diff|"]

    for i, ax in enumerate(axs):
        pcm = ax.pcolormesh(X, Y, fields[i],
                            cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
        ax.set_title(titles[i], fontsize=10)
        ax.set_aspect('equal')
        ax.axis('off')
        plt.colorbar(pcm, ax=ax, shrink=0.7)


# --------------------------------------------------------------------------
def plot_all_preds(pred_list, truth_list, case_names, grid_dict,
                   key, save_dir, config, uniform_scale=True,
                   best_fin='', cmap='RdBu_r'):
    """Plot each tensor component (truth/pred/diff) for all cases."""
    n_features = pred_list[0].squeeze().shape[-1]
    feature_names = get_tensor_component_names(key, config)

    # --- Determine global vmin/vmax across all cases if uniform scale ---
    if uniform_scale:
        vmins, vmaxs = [], []
        for i in range(n_features):
            all_vals = []
            for t, p in zip(truth_list, pred_list):
                all_vals.append(t.squeeze()[:, i])
                all_vals.append(p.squeeze()[:, i])
            all_vals = torch.cat(all_vals)
            max_abs = torch.max(torch.abs(all_vals)).item()
            vmins.append(-max_abs)
            vmaxs.append(max_abs)
    else:
        vmins = [None] * n_features
        vmaxs = [None] * n_features

    mode = 'global' if uniform_scale else 'rowwise'

    # --- Loop over cases ---
    for idx, case_name in enumerate(case_names):
        cx, cy = grid_dict[case_name]
        truth_tensor = truth_list[idx].squeeze().cpu().numpy()
        pred_tensor  = pred_list[idx].squeeze().cpu().numpy()

        if truth_tensor.ndim == 1:
            truth_tensor = truth_tensor[:, np.newaxis]
        if pred_tensor.ndim == 1:
            pred_tensor = pred_tensor[:, np.newaxis]

        fig, axs = plt.subplots(n_features, 3,
                                figsize=(12, 2.5 * n_features),
                                constrained_layout=True)
        if n_features == 1:
            axs = np.expand_dims(axs, axis=0)

        for i in range(n_features):
            truth = truth_tensor[:, i]
            pred  = pred_tensor[:, i]
            name = feature_names[i]
            title_prefix = f"{key.upper()} {name}"

            if not uniform_scale:
                max_abs = max(np.abs(truth).max(), np.abs(pred).max())
                vmin, vmax = -max_abs, max_abs
            else:
                vmin, vmax = vmins[i], vmaxs[i]

            plot_triplet_pcolormesh(axs[i], cx, cy, truth, pred,
                                    title_prefix, vmin=vmin, vmax=vmax, cmap=cmap)

        fig.suptitle(f"{case_name} – {key.upper()} ({mode} scale)",
                     fontsize=14, y=0.995)
        fig_path = Path(save_dir) / f"{best_fin}_model" / f"{case_name}_{key}_{mode}_scale.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------------
def plot_all_tensor_fields(pred_dict, truth_dict, case_names, grid_dict,
                           save_dir, config, best_fin=''):
    """Wrapper to handle all tensor families (rst, a_ij, b_ij, tke)."""
    tensor_keys = ['a_ij', 'b_ij', 'rst', 'tke']

    for key in tensor_keys:
        pred_list = pred_dict[key]
        truth_list = truth_dict[key]

        # Skip empty ones
        if pred_list is None or all(p is None for p in pred_list):
            continue

        save_subdir = Path(save_dir) / key
        save_subdir.mkdir(parents=True, exist_ok=True)

        # Global (uniform) scale
        plot_all_preds(pred_list, truth_list, case_names, grid_dict,
                       key, save_subdir, config, uniform_scale=True, best_fin=best_fin)

        # Row-wise (non-uniform) scale
        plot_all_preds(pred_list, truth_list, case_names, grid_dict,
                       key, save_subdir, config, uniform_scale=False, best_fin=best_fin)
