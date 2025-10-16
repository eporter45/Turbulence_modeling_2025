# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 09:46:32 2025

@author: eoporter
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import gc

def plot_input_kde(x_train_normed, x_test_normed, train_cases, test_cases, config, save_dir):
    """
    Plot KDE for each input feature from normalized x_train/x_test.

    Args:
        x_train_normed (list of tensors): Normalized training inputs (one per case).
        x_test_normed (list of tensors): Normalized test inputs (one per case).
        train_cases (list): Names of training cases.
        test_cases (list): Names of test cases.
        config (dict): Simulation config.
        save_dir (str or Path): Directory where plots will be saved.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    input_feature_names = config['features']['input']
    print(f"[DEBUG] Input feature count: {x_train_normed[0].shape[1]}")

    for feat_idx, feat_name in enumerate(input_feature_names):
        plt.figure()

        # Plot KDEs for each train case
        for case, xnorm in zip(train_cases, x_train_normed):
            vals = xnorm[:, feat_idx].cpu().numpy()
            sns.kdeplot(vals, label=f"{case} (train)", linestyle='-')

        # Plot KDEs for each test case
        for case, xnorm in zip(test_cases, x_test_normed):
            vals = xnorm[:, feat_idx].cpu().numpy()
            sns.kdeplot(vals, label=f"{case} (test)", linestyle='--')

        # Add titles, labels, legend
        plt.title(f"KDE: {feat_name}")
        plt.xlabel(f"{feat_name} (normalized)")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()

        # Save to file
        save_path = save_dir / f"KDE_{feat_name}.png"
        plt.savefig(save_path)
        print(f"[INFO] Saved plot: {save_path}")

        plt.close()
        gc.collect()


