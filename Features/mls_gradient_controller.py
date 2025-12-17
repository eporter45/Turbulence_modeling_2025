import numpy as np
import pandas as pd
from scipy.spatial import KDTree

''' at all costs, we do not want to do any interpolation on this data, 
    it skews the original and you can lose critical information about the system within
    functions like np.gradient or scipy gradient can perform grid based differntiation.
    However, you must construct a uniform grid from the unique x and y values of your mesh.
    It is tricky when one region of your mesh is more refined than another and many new points 
    will be added into your domain. This is when you have to integrate to fill the holes of your data. 
    This is where you are at the mercy of numerical methods and differentiation efforts.'''

''' As a workaround, I will be using the Moving Least Squares (MLS) gradient 
    estimation method, which is highly robust to irregularly spaced data. 
    The core idea of MLS is to locally approximate the scalar field around a 
    point of interest by fitting a linear function using its neighboring points 
    within a specified radius. 

    For each point, we:
      - Identify neighboring points within a radius using a spatial tree (e.g., KDTree).
      - Fit a plane (in 2D) or hyperplane (in higher dimensions) of the form:
            f(x, y) ≈ a + b*(x - x₀) + c*(y - y₀)
        where (x₀, y₀) is the point of interest.
      - Solve a weighted least squares problem to determine the coefficients.
        The weights decay with distance — typically using a Gaussian kernel.
      - The coefficients (b, c) correspond to the gradient ∇f at that point.

    This approach avoids interpolation by working directly on the original mesh, 
    respects local data density. It is only one step as well, so instead of interpolating
    and then taking the gradient, two steps where lots of information can be lost, we perform
    this method to reduce the amount of deviation from the true data and reduce errors that we could introduce
    This is useful for the model because misrepresenting the data can make it harder for models to converge'''
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 11:48:20 2025

@author: eoporter
"""
# ---------------------------------------------------------------
# === Core MLS Kernel ===========================================
# ---------------------------------------------------------------
def mls_gradient2d(coords, values, radius, kernel='gaussian', eps=1e-8):
    """
    Estimate gradients using Moving Least Squares (MLS) in 2D.

    Parameters
    ----------
    coords : (N, 2) array
        (x, y) coordinates.
    values : (N,) array
        Scalar field values.
    radius : float
        Local neighborhood radius.
    kernel : str
        'gaussian', 'inverse', or 'uniform'.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    grads : (N, 2) array
        Gradient estimates (df/dx, df/dy).
    """
    tree = KDTree(coords)
    grads = np.zeros_like(coords)
    N = len(coords)

    for i, (xi, yi) in enumerate(coords):
        idx = tree.query_ball_point([xi, yi], radius)
        if i % 10000 == 0 or i == N - 1:
            print(f"[MLS] Processing point {i+1} / {N}")
        if len(idx) < 3:
            grads[i] = np.nan
            continue

        neighbors = coords[idx]
        displacements = neighbors - [xi, yi]
        f_neighbors = values[idx]

        if kernel == 'gaussian':
            dists2 = np.sum(displacements**2, axis=1)
            weights = np.exp(-dists2 / (radius**2 + eps))
        elif kernel == 'inverse':
            dists = np.linalg.norm(displacements, axis=1)
            weights = 1 / (dists + eps)
        else:
            weights = np.ones(len(idx))

        A = displacements
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


# ---------------------------------------------------------------
# === Stage-specific Gradient Wrappers ==========================
# ---------------------------------------------------------------
def compute_stage_gradients(df, stage, radius=0.02, kernel='gaussian'):
    """
    Compute MLS gradients for a given feature-engineering stage.
    Stage 1 → primitive fields (Ux, Uy, Uz, T, p)
    Stage 2 → derived fields (ρUᵢUⱼ, τ_ij)
    """
    coords = df[['Cx', 'Cy']].to_numpy()
    dz = np.zeros(len(coords))  # planar assumption

    if stage == 1:
        feature_list = ['Ux', 'Uy', 'Uz', 'T', 'p']
        print(f"[Stage 1] Computing MLS gradients for {feature_list}")
    elif stage == 2:
        feature_list = [
            'rho_UxUx', 'rho_UxUy', 'rho_UxUz',
            'rho_UyUy', 'rho_UyUz', 'rho_UzUz',
            'Tao_xx', 'Tao_xy', 'Tao_xz',
            'Tao_yy', 'Tao_yz', 'Tao_zz'
        ]
        print(f"[Stage 2] Computing MLS gradients for {feature_list}")
    else:
        raise ValueError("Stage must be 1 or 2")

    for f in feature_list:
        if f not in df.columns:
            print(f"[WARN] {f} not in DataFrame, skipping.")
            continue
        grads = mls_gradient2d(coords, df[f].to_numpy(), radius, kernel)
        df[f'd{f}_dx'] = grads[:, 0]
        df[f'd{f}_dy'] = grads[:, 1]
        df[f'd{f}_dz'] = np.zeros_like(df["Cx"])  # planar assumption

    return df


# ---------------------------------------------------------------
# === Unified Gradient Controller ===============================
# ---------------------------------------------------------------
def compute_gradients_by_stage(df, stage, grad_mode='mls', radius=0.02, kernel='gaussian'):
    """
    Master controller deciding which gradients to compute.
    - OG → skip Stage 1 (already present)
    - MLS → compute with MLS kernel
    - Stage 2 always computed
    """
    if stage == 1 and grad_mode == 'og':
        print("[INFO] OG mode: skipping Stage 1 gradients (already in dataset).")
        return df

    print(f"[INFO] Computing Stage {stage} gradients ({grad_mode.upper()})")
    df = compute_stage_gradients(df, stage, radius=radius, kernel=kernel)
    return df