# ============================================================
#  loss_calc.py -- updated with scalar-safe reductions
# ============================================================

import torch
import torch.nn as nn

from utils.extract_tensors import (
    detect_family,
    get_indices,
    reconstruct_rst,
    AIJ6,
    compute_invariants_eigen,
)
from utils.torch_tensor_utils import vec6_to_sym3


# ============================================================
#  MATRIX LOSSES
# ============================================================

def frobenius_loss(R_pred, R_true):
    return torch.mean(torch.norm(R_pred - R_true, dim=(-2, -1), p='fro'))


def log_euclidean_loss(R_pred, R_true, eps=1e-12):
    R_pred = 0.5*(R_pred + R_pred.transpose(-2,-1))
    R_true = 0.5*(R_true + R_true.transpose(-2,-1))

    def mat_log(M):
        lam, V = torch.linalg.eigh(M)
        lam = torch.clamp(lam, min=eps)
        return V @ torch.diag_embed(torch.log(lam)) @ V.transpose(-2,-1)

    return torch.mean(torch.norm(mat_log(R_pred) - mat_log(R_true), dim=(-2,-1), p='fro'))


def spd_loss(R):
    lam = torch.linalg.eigvalsh(R)
    return torch.mean(torch.relu(-lam))


def riemannian_distance(R1, R2, eps=1e-12):
    R1 = 0.5*(R1 + R1.transpose(-2,-1))
    R2 = 0.5*(R2 + R2.transpose(-2,-1))

    lam, V = torch.linalg.eigh(R1)
    lam = torch.clamp(lam, min=eps)
    R1_inv_sqrt = V @ torch.diag_embed(lam.rsqrt()) @ V.transpose(-2,-1)

    M = R1_inv_sqrt @ R2 @ R1_inv_sqrt

    lam2 = torch.linalg.eigvalsh(M)
    lam2 = torch.clamp(lam2, min=eps)

    return torch.mean(torch.norm(torch.log(lam2), dim=-1))


# ============================================================
#  MAIN LOSS ASSEMBLY (family dispatcher)
# ============================================================

def compute_all_losses(config, pred, truth, criterion,
                       y_frob_max=None, y_k_max=None, epoch=0):

    feats  = config['features']['output']
    family = detect_family(feats)

    pred_d  = pred.clone()
    truth_d = truth.clone()

    # -----------------------------------------
    # OPTIONAL DENORMALIZATION
    # -----------------------------------------
    if config['features'].get('denorm_loss', False):

        if family == 'aij':
            a_idx = get_indices(AIJ6, feats)
            pred_d[:,a_idx]  *= y_frob_max
            truth_d[:,a_idx] *= y_frob_max

            if 'k' in feats or 'tke' in feats:
                k_idx = feats.index('k') if 'k' in feats else feats.index('tke')
                pred_d[:,k_idx]  *= y_k_max
                truth_d[:,k_idx] *= y_k_max

        else:  # RST
            pred_d  *= y_frob_max
            truth_d *= y_frob_max


    # -----------------------------------------
    # FAMILY DISPATCH
    # -----------------------------------------

    # -------- AIJ6 ONLY --------
    if family == 'aij' and len(feats) == 6:
        a_pred = vec6_to_sym3(pred_d)
        a_true = vec6_to_sym3(truth_d)
        return compute_losses_AIJ6(config, pred_d, truth_d, a_pred, a_true, criterion)

    # -------- RST OR AIJ+K --------
    R_pred, R_true = reconstruct_rst(pred_d, truth_d, feats)
    return compute_losses_RST(config, pred_d, truth_d, R_pred, R_true, criterion)


# ============================================================
#  FAMILY: AIJ6 LOSSES (must return scalars)
# ============================================================

def compute_losses_AIJ6(config, pred_d, truth_d, a_pred, a_true, criterion):
    losses = {}
    data_cfg = config['training']['loss']['terms']['data']
    phys_cfg = config['training']['loss']['terms']['phys']
    cons_cfg = config['training']['loss']['terms']['constraint']

    # -----------------------
    # DATA LOSSES
    # -----------------------
    if data_cfg.get("enabled", False):
        types = [t.lower() for t in data_cfg.get("types", [])]

        if "crit" in types:
            raw = criterion(pred_d, truth_d)      # (N,6)
            losses["data_crit"] = raw.mean()

        if "component" in types:
            losses["data_component"] = ((pred_d - truth_d)**2).mean()

        if "frob" in types:
            losses["data_frob"] = torch.norm(a_pred - a_true, dim=(-2, -1)).mean()

    # -----------------------
    # PHYSICS LOSSES
    # -----------------------
    if phys_cfg.get("enabled", False):
        types = [t.lower() for t in phys_cfg.get("types", [])]

        if "inv_a" in types:
            inv_pred = compute_invariants_eigen(a_pred)
            inv_true = compute_invariants_eigen(a_true)
            losses["phys_inv_a"] = ((inv_pred - inv_true)**2).mean()

        if "lumley" in types:
            lam = torch.linalg.eigvalsh(a_pred)
            losses["phys_lumley"] = torch.relu(-lam).mean()

    # -----------------------
    # CONSTRAINT LOSSES
    # -----------------------
    if cons_cfg.get("enabled", False):
        types = [t.lower() for t in cons_cfg.get("types", [])]

        if "traceless" in types:
            tr = a_pred[:,0,0] + a_pred[:,1,1] + a_pred[:,2,2]
            losses["const_tr"] = tr.abs().mean()

        if "symmetry" in types:
            losses["const_sym"] = (a_pred - a_pred.transpose(-2,-1)).abs().mean()

    return losses


# ============================================================
#  FAMILY: RST LOSSES (must return scalars)
# ============================================================

def compute_losses_RST(config, pred_d, truth_d, R_pred, R_true, criterion):
    losses = {}
    data_cfg = config['training']['loss']['terms']['data']

    # -----------------------
    # DATA LOSSES
    # -----------------------
    if data_cfg.get("enabled", False):
        types = [t.lower() for t in data_cfg.get("types", [])]

        if "crit" in types:
            raw = criterion(pred_d, truth_d)
            losses["data_crit"] = raw.mean()

        if "component" in types:
            losses["data_component"] = ((pred_d - truth_d)**2).mean()

        if "frob" in types:
            losses["data_frob"] = torch.norm(R_pred - R_true, dim=(-2,-1)).mean()

        if "log_euclidean" in types:
            losses["data_log"] = log_euclidean_loss(R_pred, R_true)

        if "riemann" in types:
            losses["data_riem"] = riemannian_distance(R_pred, R_true)

        if "spd" in types:
            losses["data_spd"] = spd_loss(R_pred)

    return losses
