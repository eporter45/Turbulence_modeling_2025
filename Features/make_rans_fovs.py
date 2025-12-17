import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Dynamically resolve project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def make_rans_fovs(case, rans_dir, exp_dir, dnd='nondim', grad_type='MLS', save=True):
    """
    Generate RANS FOV slices aligned to experimental FOVs.

    Parameters
    ----------
    case : str
        Case identifier, e.g. 'Case2' or 'bump_h31'
    rans_dir : str or Path
        Directory containing RANS dataset(s)
    exp_dir : str or Path
        Directory containing EXP dataset(s)
    dnd : str
        'nondim' or 'dim' (used for file naming)
    grad_type : str
        Gradient type: 'MLS' or 'OG'
    save : bool
        If True, save sliced FOVs to disk

    Returns
    -------
    dict
        Dictionary mapping FOV names → RANS-sliced DataFrames
    """

    # ------------------- Load RANS base --------------------------
    grad_subdir = f"{grad_type}_grads"
    rans_path = Path(rans_dir) / grad_subdir / "all_feats" / f"{case}_{dnd}_all_feats.pkl"

    if not rans_path.exists():
        raise FileNotFoundError(f"[ERROR] RANS file not found: {rans_path}")
    rans_df = pd.read_pickle(rans_path)

    # ------------------- Locate EXP FOVs --------------------------
    exp_base = Path(exp_dir) / f"{dnd}_exp"
    exp_prefix = f"{dnd}_{case}"
    fov_names = [f"FOV{i}" for i in range(1, 5)]
    slice_bounds = {}

    print(f"[INFO] Searching for EXP FOVs in: {exp_base}")

    for fov in fov_names:
        exp_path = exp_base / f"{exp_prefix}_{fov}.pkl"
        if not exp_path.exists():
            print(f"[WARN] {case} does not have {fov} ({exp_path.name} not found)")
            continue

        exp_df = pd.read_pickle(exp_path)
        if not {'Cx', 'Cy'}.issubset(exp_df.columns):
            print(f"[WARN] {fov} missing coordinate fields, skipping.")
            continue

        slice_bounds[fov] = {
            'x': (exp_df['Cx'].min(), exp_df['Cx'].max()),
            'y': (exp_df['Cy'].min(), exp_df['Cy'].max())
        }

    # --- If no FOVs found, return RANS unchanged
    if not slice_bounds:
        print(f"[INFO] No EXP FOVs found for {case}. Returning original RANS dataset.")
        return {'Full': rans_df}

    # ------------------- Slice RANS per FOV -----------------------
    rans_fovs = {}
    for fov, bds in slice_bounds.items():
        subset = rans_df[
            (rans_df['Cx'] >= bds['x'][0]) &
            (rans_df['Cx'] <= bds['x'][1]) &
            (rans_df['Cy'] >= bds['y'][0]) &
            (rans_df['Cy'] <= bds['y'][1])
            ]
        rans_fovs[fov] = subset
        print(f"[INFO] {case} | {fov} subset: {len(subset)} points")

        # Save individual FOV slice
        if save:
            save_dir = Path(rans_dir) / grad_type
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{case}_{dnd}_{fov}.pkl"
            subset.to_pickle(save_path)
            print(f"[SAVED] {fov} slice saved → {save_path}")

    return rans_fovs