import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# === Turbulence Kinetic Energy =================================
# ---------------------------------------------------------------
def make_tke(df):
    """
    Full 3-component turbulence kinetic energy.
    These come from EXP measurements (uu, vv, ww).
    """
    df['tke'] = 0.5 * (df['uu'] + df['vv'] + df['ww'])
    return df


# ---------------------------------------------------------------
# === Anisotropy Tensor a_ij ====================================
# ---------------------------------------------------------------
def make_aij(df):
    """
    a_ij = R_ij - 2/3 * k * δ_ij
    Retains full 3×3 turbulence anisotropy.
    """
    k = df['tke']
    two_th_k = (2/3) * k

    df['a_xx'] = df['uu'] - two_th_k
    df['a_yy'] = df['vv'] - two_th_k
    df['a_zz'] = df['ww'] - two_th_k

    df['a_xy'] = df['uv']
    df['a_yx'] = df['uv']

    df['a_xz'] = df['uw'] if 'uw' in df.columns else 0.0
    df['a_zx'] = df['a_xz']

    df['a_yz'] = df['vw'] if 'vw' in df.columns else 0.0
    df['a_zy'] = df['a_yz']

    return df


# ---------------------------------------------------------------
# === Normalized Anisotropy Tensor b_ij ==========================
# ---------------------------------------------------------------
def make_bij(df):
    """
    b_ij = a_ij / (2 k)
    Equivalent to R_ij/k - 1/3.
    Uses full 3×3.
    """
    k = df['tke']

    df['b_xx'] = df['a_xx'] / (2*k)
    df['b_yy'] = df['a_yy'] / (2*k)
    df['b_zz'] = df['a_zz'] / (2*k)

    df['b_xy'] = df['a_xy'] / (2*k); df['b_yx'] = df['b_xy']
    df['b_xz'] = df['a_xz'] / (2*k); df['b_zx'] = df['b_xz']
    df['b_yz'] = df['a_yz'] / (2*k); df['b_zy'] = df['b_yz']

    return df


# ---------------------------------------------------------------
# === Eigenvalue-based invariants (3×3) =========================
# ---------------------------------------------------------------
def compute_invariants(df, prefix):
    """
    Computes invariants I, II, III and ordered eigenvalues
    for a full 3×3 symmetric turbulence tensor.
    """
    I, II, III = [], [], []
    eig1, eig2, eig3 = [], [], []

    for _, row in df.iterrows():
        T = np.array([
            [row[f'{prefix}_xx'], row[f'{prefix}_xy'], row[f'{prefix}_xz']],
            [row[f'{prefix}_yx'], row[f'{prefix}_yy'], row[f'{prefix}_yz']],
            [row[f'{prefix}_zx'], row[f'{prefix}_zy'], row[f'{prefix}_zz']]
        ])

        # eigenvalues of symmetric 3×3
        vals = np.sort(np.linalg.eigvalsh(T))[::-1]

        I.append(np.trace(T))
        II.append(np.trace(T @ T))
        III.append(np.linalg.det(T))

        eig1.append(vals[0]); eig2.append(vals[1]); eig3.append(vals[2])

    df[f'{prefix}_I'] = I
    df[f'{prefix}_II'] = II
    df[f'{prefix}_III'] = III

    df[f'{prefix}_eig1'] = eig1
    df[f'{prefix}_eig2'] = eig2
    df[f'{prefix}_eig3'] = eig3

    return df


# ---------------------------------------------------------------
# === Barycentric Mapping =======================================
# ---------------------------------------------------------------
def add_barycentric_coords(df):
    """
    Uses eigenvalues of b_ij (normalized anisotropy)
    to compute barycentric coordinates (C1, C2, C3).
    """
    l1 = df['b_eig1']
    l2 = df['b_eig2']
    l3 = df['b_eig3']   # full 3×3 turbulence → real value

    df['C1'] = l1 - l2
    df['C2'] = 2 * (l2 - l3)
    df['C3'] = 3 * l3 + 1

    return df


# ---------------------------------------------------------------
# === Skewness (3rd-order moments) ==============================
# ---------------------------------------------------------------
def add_skewness(df):
    """
    Standard EXP skewness metrics.
    """
    df['Skew_u'] = df['uuu'] / (df['uu'] ** 1.5)
    df['Skew_v'] = df['vvv'] / (df['vv'] ** 1.5)
    return df


# ---------------------------------------------------------------
# === Transport Ratios ==========================================
# ---------------------------------------------------------------
def add_transport_ratio(df):
    """
    Transport-based ratios (Ling 2016-style).
    Uses triple correlations uuu, vuu, uvv, vvv.
    """
    rms_u = np.sqrt(df['uu'])
    rms_v = np.sqrt(df['vv'])

    df['TR_uuu'] = df['uuu'] / (df['uu'] * rms_u)
    df['TR_vuu'] = df['vuu'] / (df['uv'] * rms_u)
    df['TR_uvv'] = df['uvv'] / (df['uv'] * rms_v)
    df['TR_vvv'] = df['vvv'] / (df['vv'] * rms_v)

    num = df[['uuu','vuu','uvv','vvv']].sum(axis=1)
    df['TR_ijk'] = num / (df['tke'] ** 1.5)

    return df


# ---------------------------------------------------------------
# === Unified EXP feature wrapper ===============================
# ---------------------------------------------------------------
def compute_exp_postgrad_features(df, delta_U=None):
    """
    Full experimental turbulence-correlation post-processing.
    Retains 3D turbulence physics but assumes 2D mean flow.
    """

    # Reynolds stress nondimensionalization (optional)
    if delta_U is not None:
        df['Re_xx'] = df['uu']/delta_U**2
        df['Re_xy'] = df['uv']/delta_U**2
        df['Re_yy'] = df['vv']/delta_U**2
        df['Re_zz'] = df['ww']/delta_U**2

    # Core fields
    df = make_tke(df)
    df = make_aij(df)
    df = make_bij(df)

    # Eigenvalues and invariants
    df = compute_invariants(df, 'a')
    df = compute_invariants(df, 'b')

    # Mixing-state mapping (barycentric)
    df = add_barycentric_coords(df)

    # Higher-order turbulence stats
    if 'uuu' in df.columns:
        df = add_skewness(df)
    if 'vuu' in df.columns:
        df = add_transport_ratio(df)

    return df


def make_all_exp_features(df, delta_U=None):
    """Compatibility alias."""
    return compute_exp_postgrad_features(df, delta_U)
