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

def plot_input_kde(x_train_normed: object, x_test_normed: object, train_cases: object, test_cases: object, config: object, save_dir: object) -> None:
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


def plot_kdes_by_feature(x_train, x_test, norm='raw', save=False, save_path=''):
    all_data = {**x_train, **x_test}
    feature_names = list(next(iter(all_data.values())).columns)

    for feat in feature_names:
        plt.figure(figsize=(8, 4))
        plotted = False
        for case, df in all_data.items():
            if df[feat].nunique() > 1:
                sns.kdeplot(df[feat], label=case, linewidth=1.5)
                plotted = True
            else:
                print(f"[INFO] Skipping KDE for feature {feat} in case {case} due to 0 variance")

        if plotted:
            plt.title(f'KDE for {norm} Feature: {feat}')
            plt.xlabel('Normalized Value')
            plt.ylabel('Density')
            plt.legend()
            plt.tight_layout()

            if save:
                if save_path == '':
                    raise ValueError('[Error] no savepath passed into function')
                fname = f'{feat}_{norm}_case_kde.png'
                fpath = os.path.join(save_path, fname)
                plt.savefig(fpath)
                plt.close()
            else:
                plt.show()
        else:
            plt.close()  # nothing plotted, avoid showing blank plots
