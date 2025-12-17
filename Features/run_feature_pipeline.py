# run_feature_pipeline.py
"""
Unified Feature Engineering Pipeline
------------------------------------
Stages:
  1. Compute full-field RANS features
  2. Slice RANS into FOVs using EXP bounds
  3. Align / downsample RANS ↔ EXP
  4. EXP feature engineering
  5. Non-dimensionalization
  6. Save train-ready datasets (dim + nondim)
Outputs saved in:
  Data/Shear_mixing/<config_name>/
Includes metadata YAML log for reproducibility.
Author: Elliot Porter
"""

import os
import sys
import pandas as pd
import yaml
from datetime import datetime

from Features.mls_gradient_controller import compute_gradients_by_stage
from Features.make_all_rans_features import (compute_rans_grad0_features,
                                             compute_rans_grad1_features,
                                             compute_rans_grad2_features)
from Features.make_all_exp_features import compute_exp_postgrad_features
from Features.nondim_exp_rans import apply_nondimensionalization
from Features.downsample_exp_rans import align_rans_exp_pair, slice_rans_by_exp_fovs
from Data.Shear_mixing.boundary_conditions import BCs
# --- Path setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
print(f"[ROOT] {project_root}")


# ===============================================================
# Utility functions
# ===============================================================

def make_save_dir(config_name):
    """Ensure consistent save root under Data/Shear_mixing/<config_name>."""
    save_dir = os.path.join(project_root, "Data", "Shear_mixing", config_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def save_checkpoint(df, case, tag, chkpt_num, config_name):
    """Save checkpoint file with numbered stage label."""
    save_root = make_save_dir(config_name)
    save_path = os.path.join(save_root, f"{case}_chkpt{chkpt_num}_{tag}.pkl")
    df.to_pickle(save_path)
    print(f"[SAVED] {case} chkpt{chkpt_num}: {tag} → {save_path}")


def save_metadata_yaml(config, save_dir, completed_cases):
    """Save metadata about the run for reproducibility."""
    meta = {
        "config_name": config.get("name", "default_run"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cases": completed_cases,
        "fovs": config.get("fovs", []),
        "stages": {
            1: "RANS full-field feature engineering",
            2: "Slice RANS into FOVs",
            3: "Align / downsample EXP↔RANS",
            4: "EXP postgrad features",
            5: "Non-dimensionalization (full + FOV)",
            6: "Train-ready datasets (dim + nondim)"
        }
    }
    meta_path = os.path.join(save_dir, "run_metadata.yaml")
    with open(meta_path, "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    print(f"[LOGGED] Metadata saved to {meta_path}")


# ===============================================================
# Main Pipeline
# ===============================================================

import time

def run_feature_pipeline(config):
    config_name = config.get("name", "default_run")
    save_root = make_save_dir(config_name)
    completed_cases = []
    timing_log = {}  # <── add this

    for case in config["cases"]:

        # Stage 1 example:
        t0 = time.time()
        print(f"[STAGE 1] Computing RANS full-field features...")
    for case in config["cases"]:
        print(f"\n==============================")
        print(f"[START] Case: {case}")
        print(f"==============================")
        timing_log[case] = {}
        start_case = time.time()
        # --- Load full RANS + EXP ---
        rans_full = pd.read_pickle(os.path.join(config["rans_dir"], f"{case}_pre_grads.pkl"))
        exp_full = pd.read_pickle(os.path.join(config["exp_dir"], f"{case}_combined.pkl"))
        print(f"[Info 1] Pre_grads_features... {rans_full.columns.to_list()}")
        print(f"[INFO] Loaded RANS ({len(rans_full)}) and EXP ({len(exp_full)})")

        # ==============================================================
        # Stage 1 — Full-field RANS feature engineering
        # ==============================================================
        print(f"[STAGE 1] Computing RANS full-field features...")
        t0 = time.time()
        rans_full = compute_rans_grad0_features(rans_full,bcs=BCs[case] )
        rans_full = compute_gradients_by_stage(rans_full, 1, grad_mode=config.get("grad_mode", "MLS"))
        rans_full = compute_rans_grad1_features(rans_full, bcs=BCs[case] )
        rans_full = compute_gradients_by_stage(rans_full, 2)
        rans_full = compute_rans_grad2_features(rans_full)
        save_checkpoint(rans_full, case, "full_rans_engineered", 1, config_name)
        timing_log[case]["Stage1"] = round(time.time() - t0, 2)

        # ==============================================================
        # Stage 2 — Slice RANS into FOVs based on EXP bounds
        # ==============================================================
        print(f"[STAGE 2] Slicing RANS into FOVs...")
        t0 = time.time()
        rans_fovs, exp_fov_bounds = slice_rans_by_exp_fovs(rans_full, exp_full, config["fovs"], case, project_root)
        for fov, rans_df in rans_fovs.items():
            save_checkpoint(rans_df, case, f"{fov}_rans", 2, config_name)
            exp_path = os.path.join(config["exp_dir"], f"{case}_{fov}.pkl")
            exp_df = pd.read_pickle(exp_path)
            save_checkpoint(exp_df, case, f"{fov}_exp", 2, config_name)
        timing_log[case]["Stage2"] = round(time.time() - t0, 2)

        # ==============================================================
        # Stage 3 — Align / downsample (RANS→EXP linear interp)
        # ==============================================================
        print(f"[STAGE 3] Aligning RANS and EXP (linear interpolation)...")
        t0 = time.time()
        down_rans, down_exp = {}, {}
        for fov in config["fovs"]:
            exp_df = pd.read_pickle(os.path.join(config["exp_dir"], f"{case}_{fov}.pkl"))
            rans_df = rans_fovs[fov]
            rans_aligned, exp_aligned = align_rans_exp_pair(rans_df, exp_df, case, fov, mode="auto", interp=True)
            down_rans[fov] = rans_aligned
            down_exp[fov] = exp_aligned
            save_checkpoint(exp_aligned, case, f"{fov}_exp_downsampled", 3, config_name)
        timing_log[case]["Stage3"] = round(time.time() - t0, 2)

        # ==============================================================
        # Stage 4 — EXP postgrad feature engineering
        # ==============================================================
        print(f"[STAGE 4] Computing EXP postgrad features...")
        t0 = time.time()
        for fov in config["fovs"]:
            exp_engineered = compute_exp_postgrad_features(down_exp[fov])
            save_checkpoint(exp_engineered, case, f"{fov}_exp_engineered", 4, config_name)
        timing_log[case]["Stage4"] = round(time.time() - t0, 2)

        # ==============================================================
        # Stage 5 — Non-dimensionalization (full + FOV)
        # ==============================================================
        print(f"[STAGE 5] Non-dimensionalizing full & FOV datasets...")
        t0 = time.time()
        rans_nd_full = apply_nondimensionalization(rans_full, case_name=case, mode="rans")
        exp_nd_full = apply_nondimensionalization(exp_full, case_name=case, mode="exp")
        save_checkpoint(rans_nd_full, case, "full_rans_nondim", 5, config_name)
        save_checkpoint(exp_nd_full, case, "full_exp_nondim", 5, config_name)

        for fov in config["fovs"]:
            exp_nd = apply_nondimensionalization(down_exp[fov], case_name=f"{case}_{fov}", mode="exp")
            rans_nd = apply_nondimensionalization(down_rans[fov], case_name=f"{case}_{fov}", mode="rans")
            save_checkpoint(exp_nd, case, f"{fov}_exp_nondim", 5, config_name)
            save_checkpoint(rans_nd, case, f"{fov}_rans_nondim", 5, config_name)
        timing_log[case]["Stage5"] = round(time.time() - t0, 2)

        # ==============================================================
        # Stage 6 — Train-ready dataset export (dim + nondim)
        # ==============================================================
        print(f"[STAGE 6] Preparing train-ready datasets...")
        t0 = time.time()
        for fov in config["fovs"]:
            exp_dim = down_exp[fov]
            rans_dim = down_rans[fov]
            exp_nd = apply_nondimensionalization(exp_dim, case_name=case, mode="exp")
            rans_nd = apply_nondimensionalization(rans_dim, case_name=case, mode="rans")

            save_checkpoint(rans_dim, case, f"train_{fov}_rans_dim", 6, config_name)
            save_checkpoint(rans_nd, case, f"train_{fov}_rans_nondim", 6, config_name)
            save_checkpoint(exp_dim, case, f"train_{fov}_exp_dim", 6, config_name)
            save_checkpoint(exp_nd, case, f"train_{fov}_exp_nondim", 6, config_name)

        timing_log[case]["Stage6"] = round(time.time() - t0, 2)
        completed_cases.append(case)
        print(f"[DONE] ✅ Completed all six checkpoints for {case}")

    # ==============================================================
    # Save metadata YAML log
    # ==============================================================
    save_metadata_yaml(config, save_root, completed_cases)
    # Add timing info to metadata
    meta_path = os.path.join(save_root, "run_metadata.yaml")
    meta = {
        "config_name": config_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cases": completed_cases,
        "timing_log": timing_log,  # <── include here
    }
    with open(meta_path, "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    print(f"[LOGGED] Metadata with timings saved to {meta_path}")

# ===============================================================
# --------------------------- CONFIG -----------------------------
# ===============================================================
if __name__ == "__main__":
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d")
    grad_mode = 'og'
    config = {
        "name": f"FEPL_{grad_mode}_{timestamp}",
        "cases": ["Case1", "Case2"],
        "fovs": ["FOV1", "FOV2", "FOV3", "FOV4"],
        "rans_dir": os.path.join(project_root, "Data", "Shear_mixing", "EX_RANS", "cleaned"),
        "exp_dir": os.path.join(project_root, "Data", "Shear_mixing", "EXP_ex"),
        "grad_mode": grad_mode,
        "interp": True,
        "nondim": True,
    }

    run_feature_pipeline(config)
