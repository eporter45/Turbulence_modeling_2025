"""
@eoporter
Utilities for extracting and reconstructing RST, AIJ, BIJ, and k
from model output tensors.

This module centralizes all indexing, family detection, extraction,
and reconstruction logic for use across the ML turbulence pipeline.
"""

import torch

# ------------------------------------------------------------
# Canonical ordering for 6-component symmetric tensors
# ------------------------------------------------------------

RST6 = ['uu','uv','vv','uw','vw','ww']
AIJ6 = ['a_xx','a_xy','a_yy','a_xz','a_yz','a_zz']
BIJ6 = ['b_xx','b_xy','b_yy','b_xz','b_yz','b_zz']


# ------------------------------------------------------------
# Family detection
# ------------------------------------------------------------

def detect_family(features):
    """
    Determine which output family the network is predicting:
        'rst'  → Reynolds stresses
        'aij'  → anisotropy only
        'aijk' → anisotropy + k/tke
        'bij'  → normalized anisotropy
    """
    f = [n.lower() for n in features]

    # anisotropy (AIJ6)
    if all(t in f for t in AIJ6):
        return 'aijk' if ('k' in f or 'tke' in f) else 'aij'

    # normalized anisotropy
    if any(n.startswith('b_') for n in f):
        return 'bij'

    # direct RST family
    if any(n in f for n in RST6):
        return 'rst'

    return 'unknown'


# ------------------------------------------------------------
# Indexing utilities
# ------------------------------------------------------------

def get_indices(targets, features):
    """Return the indices of a target list of feature names."""
    idxs = []
    for t in targets:
        try:
            idxs.append(features.index(t))
        except ValueError:
            raise ValueError(f"Feature '{t}' not found in {features}")
    return idxs


# ------------------------------------------------------------
# vec6 → symmetric 3×3 matrix
# ------------------------------------------------------------
from utils.torch_tensor_utils import vec6_to_sym3

# ------------------------------------------------------------
# Extract helpers (RST, AIJ, AIJ+k)
# ------------------------------------------------------------

def extract_rst(tensor, output_features):
    idx = get_indices(RST6, output_features)
    return tensor[:, idx], [output_features[i] for i in idx]


def extract_aij(tensor, output_features):
    idx = get_indices(AIJ6, output_features)
    return tensor[:, idx], [output_features[i] for i in idx]


def has_tke(output_features):
    names = [n.lower() for n in output_features]
    return ('tke' in names) or ('k' in names)


def tke_index(output_features):
    names = [n.lower() for n in output_features]
    if 'tke' in names:
        return names.index('tke')
    if 'k' in names:
        return names.index('k')
    raise ValueError("No tke or k found in output features.")


def extract_aij_and_k(tensor, output_features):
    """Extract a_ij and optional k/tke from model outputs."""
    a_idx = get_indices(AIJ6, output_features)
    aij = tensor[:, a_idx]

    k = None
    if has_tke(output_features):
        ki = tke_index(output_features)
        k = tensor[:, ki]

    return aij, k


# ------------------------------------------------------------
# Full R_ij reconstruction
# ------------------------------------------------------------

def reconstruct_rst(pred, truth, output_features):
    """
    Build full 3×3 Reynolds stress tensors from:
        (1) Direct RST6 predictions, or
        (2) AIJ + k/tke predictions.

    Returns:
        R_pred : [N,3,3]
        R_true : [N,3,3]
    """
    names = [n.lower() for n in output_features]

    # ---- Case 1: Direct RST prediction ----
    if all(f in names for f in RST6):
        idx = [names.index(f) for f in RST6]
        R_pred = vec6_to_sym3(pred[:, idx])
        R_true = vec6_to_sym3(truth[:, idx])
        return R_pred, R_true

    # ---- Case 2: a_ij + k → R_ij ----
    if all(f in names for f in AIJ6) and ('k' in names or 'tke' in names):
        a_idx = [names.index(f) for f in AIJ6]
        k_idx = names.index('tke') if 'tke' in names else names.index('k')

        A_pred = vec6_to_sym3(pred[:, a_idx])
        A_true = vec6_to_sym3(truth[:, a_idx])
        k_pred, k_true = pred[:, k_idx], truth[:, k_idx]

        eye = torch.eye(3, device=pred.device, dtype=pred.dtype)

        R_pred = 2.0 * k_pred[:, None, None] * (A_pred + eye/3.0)
        R_true = 2.0 * k_true[:, None, None] * (A_true + eye/3.0)

        return R_pred, R_true

    # ---- Error ----
    raise ValueError("Output features must be RST6 or AIJ6+K.")

# ---------------------------------------------------------------------
#       TKE, AIJ, BIJ, and Invariant Computations (vec6-compatible)
# ---------------------------------------------------------------------

import torch
from utils.torch_tensor_utils import vec6_to_sym3


# ============================================================
# 1. Compute TKE from Reynolds stresses
# ============================================================

def compute_k(Y, output_features):
    """
    Compute turbulent kinetic energy:
        k = 0.5 * (uu + vv + ww)

    Y : [N, 6] or [N, ?] array-like
    """
    # find diagonal order
    idx_xx = output_features.index("uu") if "uu" in output_features else 0
    idx_yy = output_features.index("vv") if "vv" in output_features else 3
    idx_zz = output_features.index("ww") if "ww" in output_features else 5

    uu = Y[:, idx_xx]
    vv = Y[:, idx_yy]
    ww = Y[:, idx_zz]

    return 0.5 * (uu + vv + ww)


# ============================================================
# 2. Compute anisotropy tensor a_ij from R_ij
# ============================================================

def compute_aij(Y, k, output_features):
    """
    Compute anisotropy tensor:
        a_ij = R_ij / (2k) - (1/3) δ_ij

    Y : [N, 6] Reynolds stress components
    k : [N] turbulent kinetic energy
    """
    R = vec6_to_sym3(Y)               # [N,3,3]
    denom = (2.0 * k).view(-1, 1, 1)  # broadcast

    a = R / (denom + 1e-12)
    I = torch.eye(3, device=Y.device).expand_as(a)
    a = a - (1.0 / 3.0) * I
    return a


# ============================================================
# 3. BIJ computation (same as a_ij definition)
# ============================================================

def compute_bij(a, k=None, output_features=None):
    """
    BIJ is exactly anisotropy tensor a_ij.
    Included for compatibility with legacy TBNN literature.
    """
    return a


# ============================================================
# 4. Invariants from vec6 anisotropy representation
# ============================================================

def compute_invariants_vec6(a_vec6):
    """
    Compute invariants (I2, I3) from vec6 anisotropy.

    a_vec6: [N, 6] → anisotropy in Voigt-like vec6
    returns:
        I1 = 0 (always)
        I2 = -0.5 * tr(a^2)
        I3 = (1/3) * tr(a^3)
    """
    A = vec6_to_sym3(a_vec6)  # [N,3,3]

    # tr(A^2)
    A2 = torch.matmul(A, A)
    trA2 = A2.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

    # tr(A^3)
    A3 = torch.matmul(A2, A)
    trA3 = A3.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

    I1 = torch.zeros_like(trA2)
    I2 = -0.5 * trA2
    I3 = (1.0 / 3.0) * trA3

    return I1, I2, I3


# ============================================================
# 5. Invariants from eigenvalues (alternate route)
# ============================================================

def compute_invariants_eigen(a_vec6):
    """
    Compute anisotropy invariants via eigenvalues.
    Useful for realizability / Lumley triangle analyses.
    """
    A = vec6_to_sym3(a_vec6)
    eigs = torch.linalg.eigvalsh(A)  # [N,3]

    λ1 = eigs[:, 0]
    λ2 = eigs[:, 1]
    λ3 = eigs[:, 2]

    I1 = λ1 + λ2 + λ3
    I2 = -(λ1 * λ2 + λ2 * λ3 + λ3 * λ1)
    I3 = λ1 * λ2 * λ3

    return I1, I2, I3
