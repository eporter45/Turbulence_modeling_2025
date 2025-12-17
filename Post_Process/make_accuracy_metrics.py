# -*- coding: utf-8 -*-
"""
Updated accuracy metrics for RST / AIJ / BIJ predictions
Using new tensor utility modules
@author: eoporter
"""

import os
import json
import torch
import numpy as np

# === NEW CONSOLIDATED UTILS ===
from utils.extract_tensors import (
    detect_family,
    reconstruct_rst,
    extract_aij_and_k,
    compute_k,
    compute_aij,
    compute_bij,
    compute_invariants_vec6,
    compute_invariants_eigen,
    RST6, AIJ6, BIJ6,
)
from utils.torch_tensor_utils import vec6_to_sym3


# ---------------------------------------------------------------------
#                     TKE RECOVERY
# ---------------------------------------------------------------------

def compute_tke_recovery(pred_list, truth_list, output_features):
    """
    TKE recovery for:
      • RST family → compute from uu+vv+ww
      • AIJ+k family → use explicit k/tke channel
    """
    family = detect_family(output_features)

    k_true_total, k_pred_total = 0.0, 0.0

    if family == "rst":
        for yp, yt in zip(pred_list, truth_list):
            k_true_total += compute_k(yt, output_features).sum().item()
            k_pred_total += compute_k(yp, output_features).sum().item()

    elif family == "aijk":
        # explicit k/tke component
        k_idx = output_features.index("tke") if "tke" in output_features else output_features.index("k")
        for yp, yt in zip(pred_list, truth_list):
            k_true_total += yt[:, k_idx].sum().item()
            k_pred_total += yp[:, k_idx].sum().item()

    else:
        return 0.0

    return 100.0 * k_pred_total / (k_true_total + 1e-12)


# ---------------------------------------------------------------------
#                     INVARIANT ACCURACY
# ---------------------------------------------------------------------

def compute_invariant_accuracy(pred_list, truth_list, output_features, family, thresholds=[0.01, 0.05]):
    """
    Percent of points where |I_pred - I_true| / |I_true| < threshold.
    Works for RST and AIJ families.
    """
    results = {f"invariant_within_{int(t*100)}pct": 0 for t in thresholds}
    total_pts = 0

    for yp, yt in zip(pred_list, truth_list):

        if family == "rst":
            # convert to anisotropy
            Rp, Rt = reconstruct_rst(yp, yt, output_features)
            k_t = compute_k(yt, output_features)
            k_p = compute_k(yp, output_features)

            a_t = compute_aij(yt, k_t, output_features)
            a_p = compute_aij(yp, k_p, output_features)

        elif family == "aij":
            a_t = yt[:, :6]
            a_p = yp[:, :6]

        else:
            raise ValueError(f"Invariant accuracy not supported for family '{family}'")

        # invariants (vec-based)
        I1_t, I2_t, I3_t = compute_invariants_vec6(a_t)
        I1_p, I2_p, I3_p = compute_invariants_vec6(a_p)

        # pack
        It = torch.stack([I1_t, I2_t, I3_t], dim=-1)
        Ip = torch.stack([I1_p, I2_p, I3_p], dim=-1)

        err = torch.abs((Ip - It) / (It + 1e-12))
        total_pts += err.numel()

        for t in thresholds:
            results[f"invariant_within_{int(t*100)}pct"] += (err < t).sum().item()

    for t in thresholds:
        results[f"invariant_within_{int(t*100)}pct"] = (
            100.0 * results[f"invariant_within_{int(t*100)}pct"] / total_pts
        )

    return results


# ---------------------------------------------------------------------
#                     REALIZABILITY CHECK
# ---------------------------------------------------------------------

def compute_realizability(pred_list, output_features, family):
    """
    Lumley triangle realizability check:
    eigenvalues of anisotropy must satisfy:
       λ_min >= -1/3   and   λ_max <= 2/3
    """
    total = 0
    real = 0

    for yp in pred_list:

        if family == "rst":
            k_p = compute_k(yp, output_features)
            a_p = compute_aij(yp, k_p, output_features)

        elif family == "aij":
            a_p = yp[:, :6]

        else:
            raise ValueError(f"Realizability check not supported for family '{family}'")

        # build anisotropy matrices
        A = vec6_to_sym3(a_p)

        eigs = torch.linalg.eigvalsh(A)
        mask = (eigs.min(dim=1).values >= -1/3.0 - 1e-6) & \
               (eigs.max(dim=1).values <=  2/3.0 + 1e-6)

        real += mask.sum().item()
        total += eigs.shape[0]

    return 100.0 * real / (total + 1e-12)


# ---------------------------------------------------------------------
#                     FROBENIUS RECOVERY
# ---------------------------------------------------------------------

def compute_tensor_recovery(pred_list, truth_list, output_features, family):
    """
    % recovery using Frobenius norms:
      • RST → compute Frobenius(R_pred) / Frobenius(R_true)
      • AIJ → compute Frobenius(a_pred) / Frobenius(a_true)
    """
    frob_true, frob_pred = 0.0, 0.0

    if family == "rst":

        for yp, yt in zip(pred_list, truth_list):
            Rp, Rt = reconstruct_rst(yp, yt, output_features)
            frob_pred += torch.norm(Rp, dim=(-2, -1), p='fro').sum().item()
            frob_true += torch.norm(Rt, dim=(-2, -1), p='fro').sum().item()

    elif family == "aij":

        for yp, yt in zip(pred_list, truth_list):
            a_p = yp[:, :6]
            a_t = yt[:, :6]
            frob_pred += torch.norm(a_p, dim=1, p='fro').sum().item()
            frob_true += torch.norm(a_t, dim=1, p='fro').sum().item()

    return 100.0 * frob_pred / (frob_true + 1e-12)


# ---------------------------------------------------------------------
#                     EVALUATION WRAPPER
# ---------------------------------------------------------------------

def evaluate_cases(pred_list, truth_list, case_names, config, output_dir, fin_best):
    """
    Runs all accuracy metrics and saves JSON.
    Unified for RST / AIJ(+k) / BIJ.
    """
    feats = config["features"]["output"]
    family = detect_family(feats)
    preds = None
    if config["model"]["type"] == "tbnn":
        preds = pred_list['tensor']
        pred_list = preds
    metrics = {}
    print(f"[INFO] Evaluating accuracy for family '{family}'")

    # --- TKE Recovery ---
    if config['model']['type'] != 'tbnn':
        metrics["tke_recovery_pct"] = compute_tke_recovery(pred_list, truth_list, feats)

    # --- Frobenius recovery ---
    metrics[f"{family}_recovery_pct"] = compute_tensor_recovery(pred_list, truth_list, feats, family)

    # --- Only RST and AIJ have meaningful invariants and realizability ---
    if family in ["rst", "aij"]:
        metrics.update(compute_invariant_accuracy(pred_list, truth_list, feats, family))
        metrics["realizable_pct"] = compute_realizability(pred_list, feats, family)

    # --- Save ---
    save_path = os.path.join(output_dir, f"accuracy_metrics_{fin_best}.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[INFO] Saved accuracy metrics → {save_path}")
    return metrics
