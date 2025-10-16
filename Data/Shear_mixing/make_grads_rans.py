# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 11:48:20 2025

@author: eoporter
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.tri as tri


#Dynamically resolve project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add to system path if not already
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(PROJECT_ROOT)
from Shear_mixing.boundary_conditions import BCs

def get_file_starting_with(dir_path, startswith=''):
    for entry in os.listdir(dir_path):
        full_path = os.path.join(dir_path, entry)
        if (
            os.path.isfile(full_path)
            and entry.startswith(startswith)
            and not entry.startswith("._")
        ):
            return full_path
    raise FileNotFoundError(f"No valid file starting with '{startswith}' found in {dir_path}")



idx = '1'
case=f'Case{idx}'
data_path = dir_path = os.path.join(PROJECT_ROOT, 'Shear_mixing', 'RANS')
dir_path = os.path.join(data_path, 'Clean')
load_path = get_file_starting_with(dir_path, case)
df = pd.read_pickle(load_path)

print(df.columns)



def mls_gradient2d(coords, values, radius, kernel='gaussian', eps=1e-8):
    """
    Estimate gradients using Moving Least Squares (MLS) in 2D.
    
    Parameters:
    - coords: (N, 2) array of (x, y) coordinates.
    - values: (N,) array of scalar field values.
    - radius: float, radius for local neighborhood.
    - kernel: str, type of weight kernel ('gaussian' or 'inverse').
    - eps: small constant to prevent division by zero.
    
    Returns:
    - grads: (N, 2) array of gradient vectors (df/dx, df/dy)
    """
    tree = KDTree(coords)
    grads = np.zeros_like(coords)
    N = len(coords)

    for i, (xi, yi) in enumerate(coords):
        idx = tree.query_ball_point([xi, yi], radius)
        if i % 500 == 0 or i == N - 1:
            print(f"[MLS] Processing point {i+1} / {N}")
        if len(idx) < 3:
            grads[i] = np.nan  # not enough neighbors to fit plane
            continue

        neighbors = coords[idx]  # (M, 2)
        displacements = neighbors - [xi, yi]  # (M, 2)
        f_neighbors = values[idx]  # (M,)

        if kernel == 'gaussian':
            dists2 = np.sum(displacements**2, axis=1)
            weights = np.exp(-dists2 / (radius**2 + eps))
        elif kernel == 'inverse':
            dists = np.linalg.norm(displacements, axis=1)
            weights = 1 / (dists + eps)
        else:
            weights = np.ones(len(idx))

        # Design matrix: [dx_j, dy_j]
        A = displacements
        W = np.diag(weights)
        b = f_neighbors - f_neighbors.mean()

        # Solve weighted least squares: (A^T W A) x = A^T W b
        try:
            ATA = A.T @ W @ A
            ATb = A.T @ W @ b
            grad = np.linalg.solve(ATA, ATb)
            grads[i] = grad
        except np.linalg.LinAlgError:
            grads[i] = np.nan  # singular matrix, skip

    return grads

def mls_gradient_3d(coords, values, radius, kernel='gaussian', eps=1e-8):
    """
    Estimate gradients using Moving Least Squares (MLS) in 3D.
    
    Parameters:
    - coords: (N, 3) array of (x, y, z) coordinates.
    - values: (N,) array of scalar field values.
    - radius: float, radius for local neighborhood.
    - kernel: str, type of weight kernel ('gaussian' or 'inverse').
    - eps: small constant to prevent division by zero.
    
    Returns:
    - grads: (N, 3) array of gradient vectors (df/dx, df/dy, df/dz)
    """
    from scipy.spatial import KDTree

    tree = KDTree(coords)
    grads = np.zeros((len(coords), 3))

    for i, center in enumerate(coords):
        idx = tree.query_ball_point(center, radius)
        if i % 500 == 0 or i == len(coords) - 1:
            print(f"[MLS-3D] Processing point {i+1} / {len(coords)}")
        if len(idx) < 4:
            grads[i] = np.nan
            continue

        neighbors = coords[idx]
        displacements = neighbors - center
        f_neighbors = values[idx]

        if kernel == 'gaussian':
            dists2 = np.sum(displacements**2, axis=1)
            weights = np.exp(-dists2 / (radius**2 + eps))
        elif kernel == 'inverse':
            dists = np.linalg.norm(displacements, axis=1)
            weights = 1 / (dists + eps)
        else:
            weights = np.ones(len(idx))

        A = displacements  # (M, 3)
        W = np.diag(weights)
        b = f_neighbors - f_neighbors.mean()

        try:
            ATA = A.T @ W @ A
            ATb = A.T @ W @ b
            grad = np.linalg.solve(ATA, ATb)
            grads[i] = grad
        except np.linalg.LinAlgError:
            grads[i] = np.nan

    return grads


def compute_and_plot_gradients(df, feat_keys, coords, radius, case_name,
                                save=False, save_path=None, filename=None,
                                show_plots=True, save_plots=False, plot_save_dir=None):
    """
    Compute MLS gradients for selected features, plot them, and optionally save the DataFrame.
    
    Parameters:
    - df: pandas DataFrame with features and coordinates
    - feat_keys: list of column names to compute gradients for
    - coords: (N, 2) array of [x, y] coordinates
    - radius: float, neighborhood radius for MLS
    - case_name: str, boundary condition case (for printing)
    - save: bool, whether to save the resulting DataFrame
    - save_path: str, directory to save the file
    - filename: str, output file name
    - show_plots: bool, whether to show plots using tripcolor_plot
    """
    print(f"[INFO] Computing gradients for case: {case_name}")
    from Plotting.plot_single_feature import tripcolor_plot
    for feat_key in feat_keys:
        print(f"\n[INFO] Calculating gradient for: {feat_key}")
        feat = df[feat_key].values
        grad = mls_gradient2d(coords, feat, radius)

        x_grad = f'd{feat_key}_dx'
        y_grad = f'd{feat_key}_dy'
        df[x_grad] = grad[:, 0]
        df[y_grad] = grad[:, 1]

        fig_x = tripcolor_plot(df, field=x_grad, cmap='jet', show=show_plots)
        fig_y = tripcolor_plot(df, field=y_grad, cmap='jet', show=show_plots)

        if save_plots and plot_save_dir:
            os.makedirs(plot_save_dir, exist_ok=True)
            fig_x_path = os.path.join(plot_save_dir, f'{x_grad}.png')
            fig_y_path = os.path.join(plot_save_dir, f'{y_grad}.png')
            fig_x.savefig(fig_x_path, dpi=300, bbox_inches='tight')
            fig_y.savefig(fig_y_path, dpi=300, bbox_inches='tight')
            print(f"[SAVED] Plots saved to:\n - {fig_x_path}\n - {fig_y_path}")

    if save and save_path and filename:
        os.makedirs(save_path, exist_ok=True)
        out_path = os.path.join(save_path, filename)
        df.to_pickle(out_path)
        print(f"\n[SAVED] Gradient-augmented DataFrame saved to:\n{out_path}")
    
    return df





radius = BCs[case]['Reference']['x_ref']
coords = np.vstack((df['Cx'], df['Cy'])).T
#grad_feats = 
#save paths
save_path = os.path.join(data_path, 'MLS_grads')
save_filename= f'{case}_mls_grads.pkl'
plot_save_dir = os.path.join(PROJECT_ROOT, 'Output_plots', 'RANS_mixing_feats')
#now call the function and save
save_plots=False
show_plots=False
save_results=True
df = compute_and_plot_gradients(
    df=df,
    feat_keys=grad_feats,
    coords=coords,
    radius=radius,
    case_name=case_name,
    save=save_results,
    save_path=save_path,
    filename=save_filename,
    show_plots=show_plots,
    save_plots=save_plots,
    plot_save_dir=plot_save_dir
)    