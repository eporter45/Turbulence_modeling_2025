# -*- coding: utf-8 -*-
"""
Organize Reynolds Stress Tensor (RST) Outputs for Analysis

Produces structured dictionaries containing:
- R_ij (Reynolds stress)
- a_ij (anisotropy)
- b_ij (normalized anisotropy)
- k (TKE)
- barycentric coordinates (C, xy)

All tensor-family detection and extraction uses centralized utilities.
"""

import math
import torch

# ---- canonical tensor families ----
RST6 = ['uu','uv','vv','uw','vw','ww']
AIJ6 = ['a_xx','a_xy','a_yy','a_xz','a_yz','a_zz']
BIJ6 = ['b_xx','b_xy','b_yy','b_xz','b_yz','b_zz']

# ---- new utilities ----
from utils.extract_tensors import (
    detect_family,
    has_tke,
    tke_index,
    extract_rst,
    extract_aij,
    extract_aij_and_k,
    reconstruct_rst
)

from utils.torch_tensor_utils import vec6_to_sym3


# ------------------------------------------------------------
# Convenience: same as _pick but uses extract functions instead
# ------------------------------------------------------------
def _extract_vec6(pred, names, output_features):
    idx = [output_features.index(nm) for nm in names]
    return pred[:, idx]


# ------------------------------------------------------------
# a_ij → R_ij
# ------------------------------------------------------------
def reconstruct_rij_from_aij(aij, k):
    """Given a_ij (vec6) + k → full R_ij (vec6)."""
    rij = aij.clone()
    add_term = (2.0 / 3.0) * k.reshape(-1, 1)
    for d in (0, 2, 5):   # xx, yy, zz
        rij[:, d] = aij[:, d] + add_term[:, 0]
    return rij


# ------------------------------------------------------------
# a_ij → b_ij
# ------------------------------------------------------------
def compute_bij_from_aij(aij, k):
    """b_ij = a_ij / (2k) - 1/3 * I."""
    denom = (2.0 * k.reshape(-1, 1)) + 1e-12
    b = aij / denom
    for d in (0, 2, 5):
        b[:, d] -= 1.0 / 3.0
    return b


# ------------------------------------------------------------
# vec6 → barycentric coordinates
# ------------------------------------------------------------
def _barycentric_from_aij_vec6(a_vec6, k, eps=1e-12):
    """
    Input:
        a_vec6 : [N,6] anisotropy in vec6 order
        k      : [N]
    Output:
        C  : [N,3]
        xy : [N,2] barycentric coordinates
    """
    if a_vec6 is None or k is None:
        return None, None

    # vec6 → 3×3 symmetric
    A = vec6_to_sym3(a_vec6)      # [N,3,3]

    # eigenvalues of a_ij (descending)
    eig = torch.linalg.eigvalsh(A)           # ascending
    eig, _ = torch.sort(eig, descending=True)

    # convert to b_ij eigenvalues: a = 2k b
    two_k = 2.0 * k.reshape(-1, 1)
    small = (two_k.abs() < eps)

    b_eigs = torch.where(
        small,
        torch.zeros_like(eig),
        eig / (two_k + eps)
    )

    l1, l2, l3 = b_eigs[:, 0], b_eigs[:, 1], b_eigs[:, 2]

    # barycentric coords
    C1 = l1 - l2
    C2 = 2 * (l2 - l3)
    C3 = 3 * l3 + 1

    C = torch.stack([C1, C2, C3], dim=-1)
    C = torch.clamp(C, min=0)
    C /= (C.sum(-1, keepdim=True) + eps)

    # map to triangle
    x = C[:, 0]*1.0 + C[:, 2]*0.5
    y = C[:, 2]*(math.sqrt(3)/2)

    xy = torch.stack([x, y], dim=-1)
    return C, xy


# ------------------------------------------------------------
# Main organization function
# ------------------------------------------------------------
def organize_rst_results(y_preds_list, y_test_list, case_names, config):
    print('[INFO] Organizing results...')

    if config.get('debug', False):
        print("[DEBUG] Debug mode ON — returning empty structure.")
        return {'pred': {}, 'truth': {}}

    fam = config['features']['output_family']
    feats = detect_family(fam)
    have_k = has_tke(feats)
    ki = tke_index(feats) if have_k else None

    results = {
        'bary_preds': {},
        'pred':  {'rst': [], 'a_ij': [], 'b_ij': [], 'tke': []},
        'truth': {'rst': [], 'a_ij': [], 'b_ij': [], 'tke': []}
    }

    for case, (yp, yt) in enumerate(zip(y_preds_list, y_test_list)):

        # ------------------------------------------------------------------
        # CASE 1 — RST family
        # ------------------------------------------------------------------
        if fam == 'rst':
            rij_p, _ = extract_rst(yp, feats)
            rij_t, _ = extract_rst(yt, feats)

            k_p = 0.5 * (rij_p[:, 0] + rij_p[:, 2] + rij_p[:, 5])
            k_t = 0.5 * (rij_t[:, 0] + rij_t[:, 2] + rij_t[:, 5])

            a_p = rij_p.clone()
            a_t = rij_t.clone()
            for d in (0, 2, 5):
                a_p[:, d] -= (2/3)*k_p
                a_t[:, d] -= (2/3)*k_t

            b_p = compute_bij_from_aij(a_p, k_p)
            b_t = compute_bij_from_aij(a_t, k_t)

        # ------------------------------------------------------------------
        # CASE 2 — AIJ family
        # ------------------------------------------------------------------
        elif fam == 'aij':
            a_p, k_p = extract_aij_and_k(yp, feats)
            a_t, k_t = extract_aij_and_k(yt, feats)

            if k_p is not None:
                rij_p = reconstruct_rij_from_aij(a_p, k_p)
                rij_t = reconstruct_rij_from_aij(a_t, k_t)
                b_p = compute_bij_from_aij(a_p, k_p)
                b_t = compute_bij_from_aij(a_t, k_t)
            else:
                rij_p = rij_t = None
                b_p = b_t = None

        # ------------------------------------------------------------------
        # CASE 3 — BIJ family
        # ------------------------------------------------------------------
        elif fam == 'bij':
            b_p = _extract_vec6(yp, BIJ6, feats)
            b_t = _extract_vec6(yt, BIJ6, feats)

            if have_k:
                k_p = yp[:, ki]
                k_t = yt[:, ki]

                a_p = b_p.clone()
                a_t = b_t.clone()
                for d in (0, 2, 5):
                    a_p[:, d] = (b_p[:, d] + 1/3) * (2 * k_p)
                    a_t[:, d] = (b_t[:, d] + 1/3) * (2 * k_t)

                rij_p = reconstruct_rij_from_aij(a_p, k_p)
                rij_t = reconstruct_rij_from_aij(a_t, k_t)
            else:
                a_p = a_t = rij_p = rij_t = k_p = k_t = None

        else:
            raise ValueError(f"[ERROR] Unknown tensor family '{fam}'")

        # ------------------------------------------------------------------
        # Barycentric mapping
        # ------------------------------------------------------------------
        C_p, xy_p = _barycentric_from_aij_vec6(a_p, k_p if have_k else torch.ones(a_p.shape[0]))

        name = case_names[case]
        results['bary_preds'][name] = {
            'C': C_p.cpu().unsqueeze(1) if C_p is not None else None,
            'xy': xy_p.cpu().unsqueeze(1) if xy_p is not None else None,
        }

        # Store pred/truth
        for key, val_p, val_t in [
            ('rst', rij_p, rij_t),
            ('a_ij', a_p, a_t),
            ('b_ij', b_p, b_t),
            ('tke', k_p, k_t),
        ]:
            results['pred'][key].append(val_p.cpu().unsqueeze(1) if val_p is not None else None)
            results['truth'][key].append(val_t.cpu().unsqueeze(1) if val_t is not None else None)

    return results
