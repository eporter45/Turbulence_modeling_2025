# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 16:43:00 2025

@author: eoporter
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri
from pathlib import Path

def plot_debug_triplet(truth, pred, case_name, cx, cy, save_dir, feature_names=None):
    """
    Plot truth, prediction, and absolute difference side-by-side for debugging.

    Args:
        truth (np.ndarray or torch.Tensor): shape (N, n_features)
        pred (np.ndarray or torch.Tensor): shape (N, n_features)
        case_name (str): case name for title and filename
        cx, cy (np.ndarray): 1D arrays of grid coordinates for triangulation
        save_dir (str or Path): directory to save plot
        key (str): tensor key for title/filename (e.g., 'rij', 'aij', 'bij', 'k')
        feature_names (list of str, optional): names for tensor components; default is indices
    """
    if hasattr(truth, 'cpu'):
        truth = truth.cpu().numpy()
    if hasattr(pred, 'cpu'):
        pred = pred.cpu().numpy()

    n_features = truth.shape[1]
    triang = tri.Triangulation(cx, cy)
    
    if feature_names is None:
        feature_names = [f'comp{i}' for i in range(n_features)]

    for i in range(n_features):
        fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

        comp_truth = truth[:, i]
        comp_pred = pred[:, i]
        comp_diff = np.abs(comp_truth - comp_pred)

        vmin = min(comp_truth.min(), comp_pred.min())
        vmax = max(comp_truth.max(), comp_pred.max())

        # Plot truth
        pcm = axs[0].tripcolor(triang, comp_truth, shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
        axs[0].set_title(" Truth")
        axs[0].axis('off')
        fig.colorbar(pcm, ax=axs[0])

        # Plot prediction
        pcm = axs[1].tripcolor(triang, comp_pred, shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
        axs[1].set_title(" Pred")
        axs[1].axis('off')
        fig.colorbar(pcm, ax=axs[1])

        # Plot absolute difference
        pcm = axs[2].tripcolor(triang, comp_diff, shading='gouraud', cmap='inferno')
        axs[2].set_title("Abs Diff")
        axs[2].axis('off')
        fig.colorbar(pcm, ax=axs[2])

        # Save figure
        save_path = Path(save_dir) / f"{case_name}_{feature_names[i]}_debug.png"
        plt.suptitle(f"{case_name} -  {feature_names[i]} Debug")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
