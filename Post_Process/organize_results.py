# -*- coding: utf-8 -*-
"""
Organize Reynolds Stress Tensor (RST) Outputs for Analysis

Transforms lists of predicted and ground truth tensors into decomposed components:
- Reynolds stress tensor (t_ij)
- Anisotropy tensor (a_ij)
- Normalized anisotropy tensor (b_ij)
- Turbulent kinetic energy (k)

Returns these organized tensors in a structured dictionary for downstream evaluation or visualization.

Created on Wed Jul 16 11:13:55 2025

@author: eoporter
"""
import math
import torch

### Helpers
RST6 = ['uu','uv','vv','uw','vw','ww']
AIJ6 = ['a_xx','a_xy','a_yy','a_xz','a_yz','a_zz']
BIJ6 = ['b_xx','b_xy','b_yy','b_xz','b_yz','b_zz']

def detect_family(output_features):
    names = [n.lower() for n in output_features]
    if any(n.startswith('a_') for n in names): return 'aij'
    if any(n.startswith('b_') for n in names): return 'bij'
    return 'rst'

def has_tke(output_features):
    names = [n.lower() for n in output_features]
    return 'tke' in names or 'k' in names

def tke_index(output_features):
    names = [n.lower() for n in output_features]
    return names.index('tke') if 'tke' in names else names.index('k')

def _pick(pred, names, output_features):
    idx = [output_features.index(nm) for nm in names]
    return pred[:, idx]

def reconstruct_rij_from_aij(aij, k):
    """a: [N,6] canonical [xx,xy,yy,xz,yz,zz], k: [N] -> rij [N,6]"""
    rij = aij.clone()
    two_thirds_k = (2.0/3.0) * k.view(-1,1)
    # diag positions in our canonical vec6 are 0 (xx), 2 (yy), 5 (zz)
    for i in (0, 2, 5):
        rij[:, i] = aij[:, i] + two_thirds_k.squeeze(1)
    return rij

def compute_bij_from_aij(aij, k):
    """b = a/(2k) - 1/3 on the diagonal"""
    two_k = 2.0 * k.view(-1,1)
    b = aij / (two_k + 1e-12)
    for i in (0, 2, 5):
        b[:, i] -= 1.0/3.0
    return b

# --- Lumley / Barycentric helpers ---
def _vec6_to_mat3x3(v6: torch.Tensor) -> torch.Tensor:
    """
    v6 shape [N,6] in canonical order [xx, xy, yy, xz, yz, zz]
    -> symmetric [N,3,3]
    """
    N = v6.shape[0]
    M = torch.zeros((N,3,3), dtype=v6.dtype, device=v6.device)
    M[:,0,0] = v6[:,0]  # xx
    M[:,0,1] = v6[:,1]  # xy
    M[:,1,0] = v6[:,1]
    M[:,1,1] = v6[:,2]  # yy
    M[:,0,2] = v6[:,3]  # xz
    M[:,2,0] = v6[:,3]
    M[:,1,2] = v6[:,4]  # yz
    M[:,2,1] = v6[:,4]
    M[:,2,2] = v6[:,5]  # zz
    return M

def _barycentric_from_aij_vec6(a_vec6: torch.Tensor, k: torch.Tensor, eps: float = 1e-12):
    """
    Input:
      a_vec6 : [N,6] deviatoric anisotropy 'a_ij' in vec6 order [xx, xy, yy, xz, yz, zz]
      k      : [N]   turbulent kinetic energy (must be >= 0)
    Returns:
      C  : [N,3] barycentric coordinates (C1, C2, C3)
      xy : [N,2] barycentric triangle (x,y)
    Uses: a = 2k * b  =>  eigenvalues(b) = eigenvalues(a) / (2k)
    """
    if a_vec6 is None or k is None:
        return None, None

    # vec6 -> [N,3,3] symmetric
    A = _vec6_to_mat3x3(a_vec6)             # [N,3,3]

    # symmetric eigenvalues (ascending), then sort descending for (λ1 ≥ λ2 ≥ λ3)
    eig = torch.linalg.eigvalsh(A)          # [N,3], ascending
    eig, _ = torch.sort(eig, dim=-1, descending=True)  # [N,3]

    # scale by 1/(2k) to get eigenvalues of b
    two_k = 2.0 * k.view(-1, 1)             # [N,1]
    small = (two_k.abs() < eps)              # mask of near-zero k

    # safe division; when k ~ 0, set b-eigs to 0 (maps to 2C vertex after normalization below)
    b_eigs = torch.where(small, torch.zeros_like(eig), eig / (two_k + eps))

    lam1 = b_eigs[:, 0]
    lam2 = b_eigs[:, 1]
    lam3 = b_eigs[:, 2]

    # barycentric coordinates from b-eigs
    C1 = lam1 - lam2
    C2 = 2.0 * (lam2 - lam3)
    C3 = 3.0 * lam3 + 1.0

    # clamp and renormalize to ensure C1,C2,C3 ∈ [0,1], sum to 1
    C = torch.stack([C1, C2, C3], dim=-1)   # [N,3]
    C = torch.clamp(C, min=0.0)
    denom = torch.sum(C, dim=-1, keepdim=True) + 1e-12
    C = C / denom

    # vertices: 1C=(1,0), 2C=(0,0), 3C=(0.5, √3/2)
    x1, y1 = 1.0, 0.0
    x2, y2 = 0.0, 0.0
    x3, y3 = 0.5, math.sqrt(3.0) / 2.0

    x = C[:, 0]*x1 + C[:, 1]*x2 + C[:, 2]*x3
    y = C[:, 0]*y1 + C[:, 1]*y2 + C[:, 2]*y3
    xy = torch.stack([x, y], dim=-1)        # [N,2]

    return C, xy



### organize results
def organize_rst_results(y_preds_list, y_test_list, case_names, config):
    print('[INFO] organizing results')
    if config.get('debug', False):
        print("[DEBUG] Debug mode ON - skipping org_rst_results")
        return {'pred': {}, 'truth': {}}

    output_features = config['features']['output']
    fam = detect_family(output_features)
    have_k = has_tke(output_features)
    k_idx = tke_index(output_features) if have_k else None

    results = {
        'bary_preds': {},
        'pred':  {'rst': [], 'a_ij': [], 'b_ij': [], 'tke': []},
        'truth': {'rst': [], 'a_ij': [], 'b_ij': [], 'tke': [] }
    }

    # import your existing utilities if you have them
    # from ... import extract_rst, compute_k, compute_aij, compute_bij

    for i,  (y_pred, y_truth) in enumerate( zip(y_preds_list, y_test_list)):
        if fam == 'rst':
            # existing flow (unchanged)
            rij_p = _pick(y_pred, RST6, output_features)
            rij_t = _pick(y_truth, RST6, output_features)

            # k from diag
            k_p = 0.5 * (rij_p[:,0] + rij_p[:,2] + rij_p[:,5])
            k_t = 0.5 * (rij_t[:,0] + rij_t[:,2] + rij_t[:,5])

            # a = r - 2/3 k I
            a_p = rij_p.clone(); a_t = rij_t.clone()
            for i in (0,2,5):
                a_p[:, i] = rij_p[:, i] - (2.0/3.0)*k_p
                a_t[:, i] = rij_t[:, i] - (2.0/3.0)*k_t

            # b from r,k
            b_p = compute_bij_from_aij(a_p, k_p)
            b_t = compute_bij_from_aij(a_t, k_t)

        elif fam == 'aij':
            a_p = _pick(y_pred, AIJ6, output_features)
            a_t = _pick(y_truth, AIJ6, output_features)
            if have_k:
                k_p = y_pred[:, k_idx]
                k_t = y_truth[:, k_idx]
                rij_p = reconstruct_rij_from_aij(a_p, k_p)
                rij_t = reconstruct_rij_from_aij(a_t, k_t)
                b_p = compute_bij_from_aij(a_p, k_p)
                b_t = compute_bij_from_aij(a_t, k_t)
            else:
                # can’t build r or b without k
                k_p = k_t = None
                rij_p = rij_t = None
                b_p = b_t = None

        else:  # fam == 'bij' (add later as you planned)
            # Provide b; if have k, lift to a and r
            b_p = _pick(y_pred, BIJ6, output_features)
            b_t = _pick(y_truth, BIJ6, output_features)
            if have_k:
                k_p = y_pred[:, k_idx]; k_t = y_truth[:, k_idx]
                # a = 2k (b + 1/3 I)
                a_p = b_p.clone(); a_t = b_t.clone()
                for i in range(6):
                    a_p[:, i] = b_p[:, i] * (2.0 * k_p)
                    a_t[:, i] = b_t[:, i] * (2.0 * k_t)
                for i in (0,2,5):
                    a_p[:, i] = (b_p[:, i] + 1.0/3.0) * (2.0 * k_p)
                    a_t[:, i] = (b_t[:, i] + 1.0/3.0) * (2.0 * k_t)
                rij_p = reconstruct_rij_from_aij(a_p, k_p)
                rij_t = reconstruct_rij_from_aij(a_t, k_t)
            else:
                a_p = a_t = None
                rij_p = rij_t = None
                k_p = k_t = None

        #get bayesian coords
        
        # Helper to push in the exact shapes you expect downstream:
        def maybe(x): return x.detach().cpu().unsqueeze(1) if x is not None else None
        C_p, xy_p = _barycentric_from_aij_vec6(a_p, k_p)
        name=str(case_names[i])
        results['bary_preds'][name]= {'C': maybe(C_p), 'xy': maybe(xy_p)}
        
        results['pred']['rst'].append(maybe(rij_p))
        results['pred']['a_ij'].append(maybe(a_p))
        results['pred']['b_ij'].append(maybe(b_p))
        results['pred']['tke'].append(maybe(k_p) if have_k else None)

        results['truth']['rst'].append(maybe(rij_t))
        results['truth']['a_ij'].append(maybe(a_t))
        results['truth']['b_ij'].append(maybe(b_t))
        results['truth']['tke'].append(maybe(k_t) if have_k else None)

    return results
