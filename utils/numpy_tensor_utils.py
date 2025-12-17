# ============================================================
#  tensor_utils_numpy.py  --  NumPy tensor utilities
# ============================================================

import numpy as np

# ------------------------------------------------------------
# Vec6 ↔ 3×3
# ------------------------------------------------------------
def ensure_vec6(x):
    x = np.asarray(x)
    if x.shape[-1] > 6:
        return x[..., :6]
    elif x.shape[-1] == 6:
        return x
    else:
        raise ValueError(f"Expected vec6 input, got shape {x.shape}")

def vec6_to_mat33(arr6):
    """
    arr6: [N,6] = [xx,xy,xz,yy,yz,zz]
    Returns [N,3,3]
    """
    arr6 = ensure_vec6(arr6)
    mats = np.zeros((arr6.shape[0], 3, 3))
    xx, xy, xz, yy, yz, zz = arr6.T
    mats[:,0,0] = xx
    mats[:,0,1] = mats[:,1,0] = xy
    mats[:,0,2] = mats[:,2,0] = xz
    mats[:,1,1] = yy
    mats[:,1,2] = mats[:,2,1] = yz
    mats[:,2,2] = zz
    return mats


def mat33_to_vec6(T):
    """Inverse of vec6_to_mat33."""
    return np.stack([
        T[...,0,0], T[...,0,1], T[...,0,2],
        T[...,1,1], T[...,1,2], T[...,2,2],
    ], axis=-1)

# ------------------------------------------------------------
# Symmetry / Deviatoric
# ------------------------------------------------------------

def symmetrize_numpy(T):
    return 0.5 * (T + np.swapaxes(T, -1, -2))

def make_traceless_numpy(T):
    tr = np.trace(T, axis1=-2, axis2=-1)[..., None, None]
    I = np.eye(3)[None, :, :]
    return T - (tr / 3.0) * I

# ------------------------------------------------------------
# SPD Projection (NumPy)
# ------------------------------------------------------------

def clamp_eigenvalues_numpy(T, eps=1e-8):
    """
    SPD projection for NumPy arrays.
    """
    T = symmetrize_numpy(T)
    lam, V = np.linalg.eigh(T)
    lam = np.clip(lam, eps, None)
    return (V * lam[..., None, :]) @ V.transpose(0, 2, 1)


def deviatoric_numpy(T):
    """
    Return the deviatoric part of each 3×3 tensor in T.

    T : [..., 3, 3]
    dev(T) = T - (trace(T)/3) * I

    Equivalent to: make_traceless_numpy(T), but provided as a
    dedicated function for clarity in notebooks and tensor basis code.
    """
    tr = np.trace(T, axis1=-2, axis2=-1)[..., None, None]
    I  = np.eye(3)[None, :, :]
    return T - (tr / 3.0) * I

def rst_vec6_to_mat33_numpy(v6):
    """
    Convert Reynolds-stress vec6 → symmetric 3×3 matrix.

    v6 : array [..., 6] = [xx, xy, xz, yy, yz, zz]
    Returns array [..., 3, 3]
    """
    v6 = ensure_vec6(v6)

    v6 = np.asarray(v6)

    assert v6.shape[-1] == 6, "Input must have last dimension = 6"

    T = np.zeros(v6.shape[:-1] + (3, 3), dtype=v6.dtype)

    xx, xy, xz, yy, yz, zz = [v6[..., i] for i in range(6)]

    T[..., 0, 0] = xx
    T[..., 0, 1] = xy
    T[..., 1, 0] = xy
    T[..., 0, 2] = xz
    T[..., 2, 0] = xz
    T[..., 1, 1] = yy
    T[..., 1, 2] = yz
    T[..., 2, 1] = yz
    T[..., 2, 2] = zz

    return T


def rst_mat33_to_vec6_numpy(T):
    """
    Convert symmetric 3×3 → vec6 in Reynolds-stress ordering.

    T : array [..., 3, 3]
    Returns array [..., 6] = [xx, xy, xz, yy, yz, zz]
    """
    T = np.asarray(T)
    assert T.shape[-2:] == (3, 3), "Input must be [...,3,3]"

    return np.stack([
        T[..., 0, 0],  # xx
        T[..., 0, 1],  # xy
        T[..., 0, 2],  # xz
        T[..., 1, 1],  # yy
        T[..., 1, 2],  # yz
        T[..., 2, 2],  # zz
    ], axis=-1)

def rst_vec4_to_mat33_numpy(v4):
    """
    Build a full 3×3 Reynolds-stress tensor from 4 components.

    v4 : [...,4] = [uu, uv, vv, ww]
    Returns [...,3,3] =
        [[uu, uv, 0],
         [uv, vv, 0],
         [ 0,  0, ww]]
    """
    v4 = np.asarray(v4)
    assert v4.shape[-1] == 4, "Expected [...,4] vector [uu, uv, vv, ww]"

    uu = v4[..., 0]
    uv = v4[..., 1]
    vv = v4[..., 2]
    ww = v4[..., 3]

    T = np.zeros(v4.shape[:-1] + (3, 3), dtype=v4.dtype)

    T[..., 0, 0] = uu
    T[..., 0, 1] = uv
    T[..., 1, 0] = uv
    T[..., 1, 1] = vv

    T[..., 2, 2] = ww

    return T

def enforce_2d_plane_tensor(T):
    """
    Zero out all out-of-plane components for 2D flow.
    T: array [N, 3, 3]
    """
    T = T.copy()

    # Off-diagonal z-components
    T[..., 0, 2] = 0.0
    T[..., 2, 0] = 0.0
    T[..., 1, 2] = 0.0
    T[..., 2, 1] = 0.0

    # Diagonal z-component
    T[..., 2, 2] = 0.0

    return T

def trace_numpy(T):
    return np.trace(T, axis1=-2, axis2=-1)