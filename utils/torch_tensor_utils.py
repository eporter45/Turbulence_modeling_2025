# ============================================================
#  tensor_utils_torch.py  --  PyTorch tensor utilities
# ============================================================

import torch

# ------------------------------------------------------------
# Symmetrization / Traceless / SPD Projection
# ------------------------------------------------------------

def symmetrize(T):
    """Enforce symmetry: T = 0.5*(T + Tᵀ)."""
    return 0.5 * (T + T.transpose(-2, -1))


def make_traceless(T):
    """Enforce trace-free constraint: T_ij -> T_ij - (tr/3) δ_ij."""
    tr = T.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]
    I = torch.eye(3, device=T.device, dtype=T.dtype).expand_as(T)
    return T - (tr / 3.0) * I


def clamp_eigenvalues(T, eps=1e-8):
    """
    SPD projection via eigenvalue clamping.
    Ensures λ_i >= eps.
    """
    T = symmetrize(T)
    lam, V = torch.linalg.eigh(T)
    lam_clamped = torch.clamp(lam, min=eps)
    return V @ torch.diag_embed(lam_clamped) @ V.transpose(-2, -1)


def project_spd_cholesky(T, eps=1e-8):
    """
    Alternative SPD projection using Cholesky reconstruction.
    Forces T = L Lᵀ with small diagonal jitter.
    """
    T = symmetrize(T)
    jitter = eps * torch.eye(3, device=T.device, dtype=T.dtype)
    T_adj = T + jitter
    L = torch.linalg.cholesky(T_adj)
    return L @ L.transpose(-2, -1)

# ------------------------------------------------------------
# Vec6 ↔ 3×3
# ------------------------------------------------------------

def vec6_to_sym3(vec6):
    """
    vec6 = [xx, xy, xz, yy, yz, zz]
    Returns [N,3,3] symmetric matrices.
    """
    xx, xy, xz, yy, yz, zz = torch.unbind(vec6, dim=-1)
    return torch.stack([
        torch.stack([xx, xy, xz], dim=-1),
        torch.stack([xy, yy, yz], dim=-1),
        torch.stack([xz, yz, zz], dim=-1),
    ], dim=-2)


def sym3_to_vec6(T):
    """Flatten 3×3 symmetric matrices → [xx,xy,xz,yy,yz,zz]."""
    return torch.stack([
        T[...,0,0], T[...,0,1], T[...,0,2],
        T[...,1,1], T[...,1,2], T[...,2,2]
    ], dim=-1)



def rst_vec6_to_mat33_torch(v6):
    """
    Convert Reynolds-stress vec6 → symmetric 3×3 tensor.

    v6 : tensor [..., 6] = [xx, xy, xz, yy, yz, zz]
    Returns tensor [..., 3, 3]
    """
    assert v6.shape[-1] == 6, "Input must have last dimension = 6"

    xx = v6[..., 0]
    xy = v6[..., 1]
    xz = v6[..., 2]
    yy = v6[..., 3]
    yz = v6[..., 4]
    zz = v6[..., 5]

    T = torch.zeros(v6.shape[:-1] + (3, 3), dtype=v6.dtype, device=v6.device)

    T[..., 0, 0] = xx
    T[..., 0, 1] = xy; T[..., 1, 0] = xy
    T[..., 0, 2] = xz; T[..., 2, 0] = xz
    T[..., 1, 1] = yy
    T[..., 1, 2] = yz; T[..., 2, 1] = yz
    T[..., 2, 2] = zz

    return T


def rst_mat33_to_vec6_torch(T):
    """
    Convert 3×3 symmetric tensor → Reynolds-stress vec6.

    T : tensor [..., 3, 3]
    Returns tensor [..., 6]
    """
    assert T.shape[-2:] == (3, 3), "Input must be [...,3,3]"

    return torch.stack([
        T[..., 0, 0],  # xx
        T[..., 0, 1],  # xy
        T[..., 0, 2],  # xz
        T[..., 1, 1],  # yy
        T[..., 1, 2],  # yz
        T[..., 2, 2],  # zz
    ], dim=-1)

def trace_torch(T: torch.Tensor) -> torch.Tensor:
    """
    Compute trace for a batch of 3×3 tensors.
    T: [N, 3, 3]
    Returns: [N] vector of traces
    """
    return T.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
