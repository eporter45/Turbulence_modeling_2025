# -*- coding: utf-8 -*-
"""
Updated accuracy metrics for RST predictions
@author: eoporter
"""

import torch
import numpy as np
from train_utils.helpers import reconstruct_rst, RST6
from train_utils.calc_data_phys_const_losses import (
    compute_k, compute_aij, compute_bij, compute_invariants_tensor
)
from train_utils.helpers import detect_family, tke_index


def compute_tke_recovery(pred_list, truth_list, output_features):
    """
    Compute % recovery of TKE depending on output family:
    - RST family: compute from uu+vv+ww.
    - AIJ(+k) family: compute directly from the k/tke channel.
    """
    family = detect_family(output_features)
    k_true_tot, k_pred_tot = 0.0, 0.0

    if family == "rst":
        # From diagonal Reynolds stresses
        for yp, yt in zip(pred_list, truth_list):
            k_true_tot += compute_k(yt, output_features).sum().item()
            k_pred_tot += compute_k(yp, output_features).sum().item()
        return 100.0 * k_pred_tot / (k_true_tot + 1e-12)
    elif family == "aij":
        # From explicit k/tke column
        if not any(f.lower() in ["k", "tke"] for f in output_features):
            raise ValueError("[ERROR] No k/tke channel in AIJ outputs for TKE recovery.")
        k_idx = tke_index(output_features)
        for yp, yt in zip(pred_list, truth_list):
            k_true_tot += yt[:, k_idx].sum().item()
            k_pred_tot += yp[:, k_idx].sum().item()
        return 100.0 * k_pred_tot / (k_true_tot + 1e-12)
    else:
        raise ValueError(f"[ERROR] TKE recovery not supported for family '{family}'")

def compute_invariant_accuracy(pred_list, truth_list, output_features, family, thresholds=[0.01, 0.05]):
    """Percent of points where invariant errors are within thresholds, supports both RST and AIJ families."""
    results = {f"invariant_within_{int(t*100)}pct": 0 for t in thresholds}
    total_points = 0

    for yp, yt in zip(pred_list, truth_list):
        if family == "rst":
            Rp, Rt = reconstruct_rst(yp, yt, output_features)
            # convert to anisotropy tensor
            k_true = compute_k(yt, output_features)
            k_pred = compute_k(yp, output_features)
            a_true = compute_aij(yt, k_true, output_features)
            a_pred = compute_aij(yp, k_pred, output_features)
        elif family == "aij":
            # take first 6 comps (a_xx,...,a_zz), ignore k
            a_true = yt[:, :6]
            a_pred = yp[:, :6]
        else:
            raise ValueError(f"[ERROR] Family '{family}' not supported in invariant accuracy")

        I_true = compute_invariants_tensor(a_true)
        I_pred = compute_invariants_tensor(a_pred)
        for It, Ip in zip(I_true, I_pred):
            err = torch.abs((Ip - It) / (It + 1e-12))
            total_points += len(err)
            for t in thresholds:
                results[f"invariant_within_{int(t*100)}pct"] += (err < t).sum().item()

    for t in thresholds:
        results[f"invariant_within_{int(t*100)}pct"] /= total_points
        results[f"invariant_within_{int(t*100)}pct"] *= 100.0
    return results


def compute_realizability(pred_list, output_features, family):
    """Percent of points with realizable anisotropy (Lumley triangle check)."""
    count_real, count_total = 0, 0

    for yp in pred_list:
        if family == "rst":
            k = compute_k(yp, output_features)
            a = compute_aij(yp, k, output_features)
        elif family == "aij":
            # directly take anisotropy comps
            a = yp[:, :6]
            k = yp[:, -1] if yp.shape[1] > 6 else None
        else:
            raise ValueError(f"[ERROR] Family '{family}' not supported in realizability")

        b = compute_bij(a, k, output_features) if k is not None else compute_bij(a, torch.ones_like(a[:, 0]), output_features)

        eigs = torch.linalg.eigvalsh(
            torch.stack([
                torch.tensor([[b[i,0], b[i,1], b[i,3]],
                              [b[i,1], b[i,2], b[i,4]],
                              [b[i,3], b[i,4], b[i,5]]])
                for i in range(b.shape[0])
            ])
        )
        # realizable if eigenvalues within [-1/3, 2/3]
        mask = (eigs.min(dim=1).values >= -1/3.0 - 1e-6) & (eigs.max(dim=1).values <= 2/3.0 + 1e-6)
        count_real += mask.sum().item()
        count_total += eigs.shape[0]

    return 100.0 * count_real / (count_total + 1e-12)

def compute_tensor_recovery(pred_list, truth_list, output_features, family):
    """
    Compute Frobenius norm recovery depending on output family.
    - RST: full R_ij Frobenius.
    - AIJ: anisotropy tensor recovery (a_ij Frobenius).
    """
    frob_true, frob_pred = 0.0, 0.0

    if family == "rst":
        for yp, yt in zip(pred_list, truth_list):
            Rp, Rt = reconstruct_rst(yp, yt, output_features)
            frob_true += torch.norm(Rt, dim=(-2, -1), p='fro').sum().item()
            frob_pred += torch.norm(Rp, dim=(-2, -1), p='fro').sum().item()
        return 100.0 * frob_pred / (frob_true + 1e-12)

    elif family == "aij":
        for yp, yt in zip(pred_list, truth_list):
            # Already anisotropy-like outputs, just compare directly
            frob_true += torch.norm(yt, dim=1, p='fro').sum().item()
            frob_pred += torch.norm(yp, dim=1, p='fro').sum().item()
        return 100.0 * frob_pred / (frob_true + 1e-12)

    else:
        raise ValueError(f"[ERROR] Tensor recovery not supported for family '{family}'")

def evaluate_cases(pred_list, truth_list, case_names, config, output_dir, fin_best):
    """
    Outer wrapper (kept same API as before).
    Runs new accuracy metrics and saves results.
    """
    import json, os

    metrics = {}
    feats = config['features']['output']
    family = detect_family(feats)

    # --- Family-dependent metrics ---
    metrics['tke_recovery_pct'] = compute_tke_recovery(pred_list, truth_list, feats)
    metrics[f'{family}_recovery_pct'] = compute_tensor_recovery(
        pred_list, truth_list, feats, family
    )

    # --- Invariants and realizability only make sense for anisotropy tensors ---
    if family in ["rst", "aij"]:
        inv_acc = compute_invariant_accuracy(pred_list, truth_list, feats, family)
        metrics.update(inv_acc)
        metrics['realizable_pct'] = compute_realizability(pred_list, feats, family)

    # --- Save ---
    save_path = os.path.join(output_dir, f"accuracy_metrics_{fin_best}.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[INFO] Saved accuracy metrics to {save_path}")
    return metrics
