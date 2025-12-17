# ===============================================================
# downsample_exp_rans.py (with interpolation)
# ===============================================================

import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.interpolate import griddata


def downsample_to_mesh(source_df, target_df, coord_cols=['Cx', 'Cy']):
    """
    Nearest-neighbor downsampling from source_df to match target_df coords.
    """
    tree = cKDTree(source_df[coord_cols].values)
    _, idx = tree.query(target_df[coord_cols].values, k=1)
    return source_df.iloc[idx].reset_index(drop=True)


def interpolate_to_target(source_df, target_df, coord_cols=['Cx', 'Cy'], method='linear'):
    """
    Interpolate source_df onto target_df coordinate mesh using griddata.
    Supports 'linear', 'nearest', or 'cubic'.
    """
    Xs = source_df[coord_cols].values
    Xt = target_df[coord_cols].values

    interp_df = pd.DataFrame(columns=source_df.columns)
    interp_df[coord_cols] = target_df[coord_cols].values

    for col in source_df.columns:
        if col in coord_cols:
            continue
        values = source_df[col].values
        try:
            interp_vals = griddata(Xs, values, Xt, method=method)
        except Exception:
            interp_vals = griddata(Xs, values, Xt, method='nearest')
        interp_df[col] = interp_vals

    return interp_df


def align_rans_exp_pair(rans_df, exp_df, case, fov, mode='auto', interp=False):
    """
    Align RANS and EXP by downsampling or interpolation.
    """
    n_rans, n_exp = len(rans_df), len(exp_df)
    if mode == 'auto':
        mode = 'rans_to_exp' if n_rans > n_exp else 'exp_to_rans'

    if mode == 'rans_to_exp':
        print(f"[ALIGN] Downsampling RANS → EXP for {case} {fov}")
        rans_aligned = interpolate_to_target(rans_df, exp_df) if interp else downsample_to_mesh(rans_df, exp_df)
        exp_aligned = exp_df.reset_index(drop=True)
    elif mode == 'exp_to_rans':
        print(f"[ALIGN] Downsampling EXP → RANS for {case} {fov}")
        exp_aligned = interpolate_to_target(exp_df, rans_df) if interp else downsample_to_mesh(exp_df, rans_df)
        rans_aligned = rans_df.reset_index(drop=True)
    else:
        raise ValueError(f"[ERROR] Invalid mode '{mode}'")

    print(f"   RANS: {n_rans} → {len(rans_aligned)}, EXP: {n_exp} → {len(exp_aligned)}")
    return rans_aligned, exp_aligned


def run_downsample_pipeline(cases, fovs, rans_dir, exp_dir, save=False, save_dir=None, mode='auto', interp=False):
    """
    Batch align RANS and EXP pairs for multiple cases/FOVs.
    """
    rans_dir = Path(rans_dir)
    exp_dir = Path(exp_dir)
    if save and save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for case in cases:
        for fov in fovs:
            for dnd in ['dim', 'nondim']:
                rans_path = rans_dir / f"{case}_{dnd}_{fov}.pkl"
                exp_path = exp_dir / f"{dnd}_exp" / f"{dnd}_{case}_{fov}.pkl"

                if not (rans_path.exists() and exp_path.exists()):
                    print(f"[WARN] Missing files for {case} {fov} ({dnd}) — skipping.")
                    continue

                try:
                    rans_df = pd.read_pickle(rans_path)
                    exp_df = pd.read_pickle(exp_path)
                    rans_aligned, exp_aligned = align_rans_exp_pair(
                        rans_df, exp_df, case, fov, mode=mode, interp=interp
                    )

                    if save and save_dir:
                        base_name = f"{case}_{dnd}_{fov}.pkl"
                        rans_save = save_dir / f"aligned_rans_{base_name}"
                        exp_save = save_dir / f"aligned_exp_{base_name}"

                        rans_aligned.to_pickle(rans_save)
                        exp_aligned.to_pickle(exp_save)
                        print(f"[SAVED] {case} {fov} ({dnd}) → {save_dir}")

                except Exception as e:
                    print(f"[ERROR] {case} {fov} ({dnd}): {e}")

#for rans slicing
def slice_rans_by_exp_fovs(rans_full, exp_full, fovs, case, project_root):
    """
    Slice RANS field into regions that match EXP FOV x-bounds.
    Returns {fov_name: rans_df}, and dict of FOV bounds.
    """
    exp_fov_bounds = {}
    rans_fovs = {}
    for fov in fovs:
        exp_fov_path = os.path.join(project_root, "Data", "Shear_mixing", "EXP_ex", f"{case}_{fov}.pkl")
        exp_df = pd.read_pickle(exp_fov_path)
        x_min, x_max = exp_df["Cx"].min(), exp_df["Cx"].max()
        y_min, y_max = exp_df["Cy"].min(), exp_df["Cy"].max()

        exp_fov_bounds[fov] = (x_min, x_max, y_min, y_max)
        mask = (rans_full["Cx"] >= x_min) & (rans_full["Cx"] <= x_max) & (rans_full["Cy"] >= y_min) & (rans_full["Cy"] <= y_max)
        rans_fovs[fov] = rans_full.loc[mask].reset_index(drop=True)

    return rans_fovs, exp_fov_bounds