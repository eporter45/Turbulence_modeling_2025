# -*- coding: utf-8 -*-
"""
Load_TBNN.py
Clean TBNN preprocessing:
 - Build S and R matrices
 - Compute full invariant dictionary
 - Compute full tensor basis dictionary
 - Subselect invariants & basis using YAML config
 - Apply k-order normalization
 - Package output for the TBNN model
"""

import numpy as np
import pandas as pd

# -------------------------------------------------------------
#   IMPORT NOTEBOOK FUNCTIONS (WE RE-IMPLEMENT THEM HERE)
# -------------------------------------------------------------
from utils.numpy_tensor_utils import vec6_to_mat33


# ------------------ Helpers ----------------------

def dev(A):
    tr = np.trace(A, axis1=1, axis2=2)[:, None, None]
    I = np.eye(3)[None,:,:]
    return A - tr/3 * I


def dcon(A,B):
    return np.sum(A*B, axis=(1,2))


# -------------------------------------------------------------
#   Compute ALL POSSIBLE invariants (Pope/Mixed/Deviatoric)
# -------------------------------------------------------------
def compute_all_invariants(S, R):

    S2 = S @ S
    R2 = R @ R
    S3 = S2 @ S
    R3 = R2 @ R

    S_dev = dev(S)
    R_dev = dev(R)
    S2_dev = dev(S2)
    R2_dev = dev(R2)

    S_dev2 = S_dev @ S_dev
    R_dev2 = R_dev @ R_dev

    SR = S @ R
    RS = R @ S

    inv = {}

    # strain-only
    inv["tr_S2"] = np.trace(S2, axis1=1, axis2=2)
    inv["tr_S3"] = np.trace(S3, axis1=1, axis2=2)
    inv["tr_S2_dev"] = np.trace(S2_dev, axis1=1, axis2=2)
    inv["tr_S_dev2"] = np.trace(S_dev2, axis1=1, axis2=2)

    # rotation-only
    inv["tr_R2"] = np.trace(R2, axis1=1, axis2=2)
    inv["tr_R3"] = np.trace(R3, axis1=1, axis2=2)
    inv["tr_R2_dev"] = np.trace(R2_dev, axis1=1, axis2=2)
    inv["tr_R_dev2"] = np.trace(R_dev2, axis1=1, axis2=2)

    # mixed
    inv["S:R"] = dcon(S, R)
    inv["S2:R"] = dcon(S2, R)
    inv["S:R2"] = dcon(S, R2)
    inv["S2:R2"] = dcon(S2, R2)

    inv["SR:RS"] = dcon(SR, RS)
    inv["SR2:S2R"] = dcon(R @ S2, S2 @ R)

    # optional lower order
    inv["tr_S"] = np.trace(S, axis1=1, axis2=2)
    inv["tr_R"] = np.trace(R, axis1=1, axis2=2)

    return inv


# -------------------------------------------------------------
#   Compute FULL tensor basis dictionary (25+ possible)
# -------------------------------------------------------------
def compute_full_tensor_basis(S,R):

    S2 = S @ S
    R2 = R @ R
    S_dev = dev(S)
    R_dev = dev(R)
    S2_dev = dev(S2)
    R2_dev = dev(R2)
    S_dev2 = S_dev @ S_dev
    R_dev2 = R_dev @ R_dev

    SR = S @ R
    RS = R @ S

    basis = {
        "S": S,
        "R": R,
        "S2": S2,
        "R2": R2,
        "S_dev": S_dev,
        "R_dev": R_dev,
        "S2_dev": S2_dev,
        "R2_dev": R2_dev,
        "S_dev2": S_dev2,
        "R_dev2": R_dev2,
        "SR": SR,
        "RS": RS,
        "SR-RS": SR - RS,
        "RS2-S2R": R @ S2 - S2 @ R,
    }

    return basis


# -------------------------------------------------------------
#   K-order normalization
# -------------------------------------------------------------
def normalize_basis_korder(basis_dict, k):
    normalized = {}
    for name, T in basis_dict.items():

        # determine polynomial order
        if name in ["S", "R"]:
            p = 1
        elif name in ["S2", "R2", "S_dev", "R_dev", "S2_dev", "R2_dev", "S_dev2", "R_dev2"]:
            p = 2
        else:
            p = 3  # mixed / commutator terms default to cubic

        # reshape k for broadcasting across 3x3 tensors
        k_b = (k ** p + 1e-12).reshape(-1, 1, 1)

        normalized[name] = T / k_b
    return normalized


# -------------------------------------------------------------
#   MAIN TBNN PREPROCESSOR
# -------------------------------------------------------------
def load_tbnn_data(cfg, raw_data_bundle):

    inv_keys = cfg['features']['invariants']
    basis_keys = cfg['features']['tensor_basis']
    output_family = cfg['features']['output_family']  # e.g., "aij", "bij", "rst"
    tbnn_data = {"train": {}, "test": {}}

    for split in ["train", "test"]:

        for case, df in raw_data_bundle[f"x_{split}"].items():

            # extract matrices
            S = df[[f"S_{ij}" for ij in ["xx","xy","xz","yx","yy","yz","zx","zy","zz"]]].to_numpy().reshape(-1,3,3)
            R = df[[f"R_{ij}" for ij in ["xx","xy","xz","yx","yy","yz","zx","zy","zz"]]].to_numpy().reshape(-1,3,3)
            k = df["k"].to_numpy().flatten()

            # compute ALL invariants
            all_inv = compute_all_invariants(S,R)

            # select requested invariants
            inv_matrix = np.column_stack([all_inv[inv] for inv in inv_keys]).astype(np.float32)

            # compute ALL basis tensors
            full_basis = compute_full_tensor_basis(S,R)

            # apply k-order normalization
            if cfg['features']['basis_norm'] == 'k_polynomial':
                full_basis = normalize_basis_korder(full_basis, k)

            # select requested basis tensors
            basis_stack = []
            for bname in basis_keys:
                B = full_basis[bname].reshape(len(S), 9)
                basis_stack.append(B)

            tensor_basis = np.stack(basis_stack, axis=1).astype(np.float32)  # (N, B, 9)

            # load anisotropy / RST / bij depending on config
            Ydf = raw_data_bundle[f"y_{split}"][case]

            # select output columns based on family type in config
            output_family = cfg['features']['output_family']
            from utils.io_tools import detect_output_cols
            out_cols = detect_output_cols(output_family)
            out_true = Ydf[out_cols].to_numpy().astype(np.float32)

            tbnn_data[split][case] = {
                "invariants": inv_matrix,
                "tensor_basis": tensor_basis,
                "out_true": out_true,
            }

    # attach grid and lumley for plotting
    from PreProcess.Load_data import load_grid_dicts
    tbnn_data["grid_dict"] = load_grid_dicts(raw_data_bundle)

    # lumley triangles
    from Plotting.plot_lumley import _exp_centerline_c123, _rans_centerline_c123
    lumley = {"train":{"RANS":{}, "EXP":{}}, "test":{"RANS":{}, "EXP":{}}}

    for split in ["train","test"]:
        for case, df in raw_data_bundle[f"x_{split}"].items():
            lumley[split]["RANS"][case] = _rans_centerline_c123(df, n=20, tol_y=1.5e-3)
        for case, df in raw_data_bundle[f"y_{split}"].items():
            lumley[split]["EXP"][case] = _exp_centerline_c123(df, n=20, tol_y=1.5e-3)

    tbnn_data["lumley_dict"] = lumley

    return tbnn_data
