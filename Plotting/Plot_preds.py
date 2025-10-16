# -*- coding: utf-8 -*-
"""
Visualization of Predicted vs. Ground Truth Tensor Fields

Provides functions to plot tensor field components (e.g., Reynolds stresses, anisotropy tensors)
for multiple cases on triangulated grids.

Features:
- Plots truth, prediction, and absolute difference side-by-side for each tensor component.
- Supports global uniform or per-sample (row-wise) color scaling.
- Automatically saves figures organized by tensor type and case.
- Uses output feature names from config to label plots appropriately.

Created on Wed Jul 16 11:04:38 2025

@author: eoporter
"""

import matplotlib.pyplot as plt
import torch
import os
from pathlib import Path
import numpy as np
import matplotlib.tri as tri


def get_tensor_component_names(key, config):
    if key == 'k':
        return ['k']
    else:
        features = config['features']['output']
        component_map = {
            'uu': '11', 'vv': '22', 'ww': '33',
            'uv': '12', 'uw': '13', 'vw': '23'
        }
        return [component_map.get(f.lower(), f) for f in features]

def plot_triplet(axs, tri, truth, pred, title_prefix, vmin=None, vmax=None, levels=100):
    diff = np.abs(truth - pred)

    # Flatten if needed
    truth = np.ravel(truth)
    pred  = np.ravel(pred)
    diff  = np.ravel(diff)

    titles = [f"{title_prefix} Truth", f"{title_prefix} Pred", f"{title_prefix} Diff"]
    values = [truth, pred, diff]

    for i, ax in enumerate(axs):
        cf = ax.tricontourf(
            tri,
            values[i],
            levels=levels,
            cmap='RdBu_r',
            vmin=vmin,
            vmax=vmax
        )
        ax.set_title(titles[i])
        ax.set_aspect('equal')
        ax.axis('off')
        plt.colorbar(cf, ax=ax, shrink=0.7)


        
        
def plot_all_preds(pred_list, truth_list, case_names, grid_dict, key, save_dir, config, uniform_scale=True, best_fin=''):
    # Squeeze example tensor to determine shape
    tensor = pred_list[0].squeeze()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(1)
    n_features = tensor.shape[1]
    feature_names = get_tensor_component_names(key, config)

    if uniform_scale:
        vmins, vmaxs = [], []
        for i in range(n_features):
            all_vals = []
            for truth_tensor, pred_tensor in zip(truth_list, pred_list):
                # Squeeze and reshape if needed
                truth_tensor = truth_tensor.squeeze()
                pred_tensor = pred_tensor.squeeze()

                if truth_tensor.ndim == 1:
                    truth_tensor = truth_tensor.reshape(-1, n_features)
                if pred_tensor.ndim == 1:
                    pred_tensor = pred_tensor.reshape(-1, n_features)

                all_vals.append(truth_tensor[:, i])
                all_vals.append(pred_tensor[:, i])
            all_vals = torch.cat(all_vals)
            abs_max = torch.max(torch.abs(all_vals))
            vmins.append(-abs_max.item())
            vmaxs.append(abs_max.item())
    else:
        vmins = [None] * n_features
        vmaxs = [None] * n_features

    mode = 'global' if uniform_scale else 'rowwise'

    for idx, case_name in enumerate(case_names):
        cx, cy = grid_dict[case_name]
        triang = tri.Triangulation(cx, cy)
        truth_tensor = truth_list[idx].squeeze()
        pred_tensor = pred_list[idx].squeeze()

        if truth_tensor.ndim == 1:
            truth_tensor = truth_tensor.reshape(-1, n_features)
        if pred_tensor.ndim == 1:
            pred_tensor = pred_tensor.reshape(-1, n_features)

        fig, axs = plt.subplots(n_features, 3, figsize=(12, 2.5 * n_features), constrained_layout=True)
        if n_features == 1:
            axs = np.expand_dims(axs, axis=0)

        for i in range(n_features):
            truth = truth_tensor[:, i].cpu().numpy()
            pred = pred_tensor[:, i].cpu().numpy()
            name = feature_names[i]
            title_prefix = f"{name}"

            if not uniform_scale:
                max_abs = max(np.abs(truth).max(), np.abs(pred).max())
                vmin = -max_abs
                vmax = max_abs
            else:
                vmin = vmins[i]
                vmax = vmaxs[i]

            plot_triplet(axs[i], triang, truth, pred, title_prefix, vmin=vmin, vmax=vmax)

        fig.suptitle(f"{case_name} - {key.upper()} ({mode} scale)", fontsize=14)

        fig_path = Path(save_dir)/f'{best_fin}_model'/ f"{case_name}_{key}_{mode}_scale.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.savefig(str(fig_path), dpi=150)
        except Exception as e:
            print(f"[ERROR] Could not save figure {fig_path}: {e}")
            raise
        plt.close(fig)


        


def plot_all_tensor_fields(pred_dict, truth_dict, case_names, grid_dict, save_dir, config, best_fin=''):
    tensor_keys = ['rst', 'a_ij', 'b_ij', 'tke']

    for key in tensor_keys:
        pred_list = pred_dict[key]
        truth_list = truth_dict[key]

        # Save in subdir per tensor key
        save_subdir = Path(save_dir) / key
        save_subdir.mkdir(parents=True, exist_ok=True)

        # Global (uniform) scale
        plot_all_preds(
            pred_list=pred_list,
            truth_list=truth_list,
            case_names=case_names,
            grid_dict=grid_dict,
            key=key,
            save_dir=save_subdir,
            config=config,
            uniform_scale=True,
            best_fin=best_fin,
        )

        # Row-wise (non-uniform) scale
        plot_all_preds(
            pred_list=pred_list,
            truth_list=truth_list,
            case_names=case_names,
            grid_dict=grid_dict,
            key=key,
            save_dir=save_subdir,
            config=config,
            uniform_scale=False,
            best_fin=best_fin
        )