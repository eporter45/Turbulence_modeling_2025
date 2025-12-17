# -*- coding: utf-8 -*-
"""
feature_engineering_rans.py

Computes derived features for RANS data in two tiers:
 - Tier 1: pre-gradient (1st order derivatives, direct physics quantities)
 - Tier 2: post-gradient (2nd derivatives, contractions, invariants, residuals)
"""

import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------
# === Tier 0: Pre-Gradient Features ============================================
# ------------------------------------------------------------------------------
def compute_rans_grad0_features(df, bcs):
    """
    Compute features that depend only on RANS primitive fields (Ux, Uy, Uz, p, T)
    and first derivatives (∂Ui/∂xj).

    Adds to df:
      - rho, mu_suth
      - strain-rate tensor S_ij
      - rotation-rate tensor R_ij
      - Q-criterion and Q_norm
      - viscous stress tensor τ_ij
      - advection tensor ρUᵢUⱼ
      - advection and strain/rotation magnitudes
    """
    # --- Density and viscosity ---
    R = 287.05
    T = df['T']
    p = df['p']
    df['rho'] = p / (R * T)

    mu_ref = bcs['Reference']['mu_ref']
    T_ref = bcs['Reference']['T_ref']
    S = 110.4
    df['mu_suth'] = mu_ref * (T / T_ref) ** 1.5 * (T_ref + S) / (T + S)
    return df


# ------------------------------------------------------------------------------
# === Tier 1: Post 1st gradiend Features ============================================
# ------------------------------------------------------------------------------
def compute_rans_grad1_features(df, bcs):
    """
    Compute features that depend only on RANS primitive fields (Ux, Uy, Uz, p, T)
    and first derivatives (∂Ui/∂xj).

    Adds to df:
      - rho, mu_suth
      - strain-rate tensor S_ij
      - rotation-rate tensor R_ij
      - Q-criterion and Q_norm
      - viscous stress tensor τ_ij
      - advection tensor ρUᵢUⱼ
      - advection and strain/rotation magnitudes
    """

    # --- Strain tensor S_ij (2D safe)---
    df['S_xx'] = 0.5 * (df['dUx_dx'] + df['dUx_dx'])
    df['S_xy'] = 0.5 * (df['dUx_dy'] + df['dUy_dx'])
    df['S_yx'] = df['S_xy']
    df['S_yy'] = 0.5 * (df['dUy_dy'] + df['dUy_dy'])
    # These MUST be zero in 2D
    df['S_xz'] = np.zeros_like(df['S_xx'])
    df['S_zx'] = df['S_xz']
    df['S_yz'] = np.zeros_like(df['S_xx'])
    df['S_zy'] = df['S_yz']
    df['S_zz'] = np.zeros_like(df['S_xx'])

    # --- Rotation tensor R_ij (2D Safe) ---
    df['R_xy'] = 0.5 * (df['dUx_dy'] - df['dUy_dx'])
    df['R_yx'] = -df['R_xy']
    # Everything involving z should be zero
    df['R_xz'] = np.zeros_like(df['R_xy'])
    df['R_zx'] = np.zeros_like(df['R_xy'])
    df['R_yz'] = np.zeros_like(df['R_xy'])
    df['R_zy'] = np.zeros_like(df['R_xy'])
    df['R_xx'] = np.zeros_like(df['R_xy'])
    df['R_yy'] = np.zeros_like(df['R_xy'])
    df['R_zz'] = np.zeros_like(df['R_xy'])

    # ============================================================
    # Frobenius norms (2-D safe)
    # ============================================================
    df['Strain_mag'] = np.sqrt(
        df['S_xx'] ** 2 + df['S_yy'] ** 2 + df['S_zz'] ** 2 +
        2 * (df['S_xy'] ** 2 + df['S_xz'] ** 2 + df['S_yz'] ** 2)
    )

    df['Rot_mag'] = np.sqrt(
        2 * (df['R_xy'] ** 2 + df['R_xz'] ** 2 + df['R_yz'] ** 2)
    )

    # ============================================================
    # Q-criterion
    # ============================================================
    S_sq = df['Strain_mag'] ** 2
    R_sq = df['Rot_mag'] ** 2

    df['Q_crit'] = 0.5 * (R_sq - S_sq)
    df['Q_norm'] = (R_sq - S_sq) / (R_sq + S_sq + 1e-12)

    # ============================================================
    # Advection tensor ρ Uᵢ Uⱼ (2-D)
    # ============================================================
    rho = df['rho']
    Ux = df['Ux']
    Uy = df['Uy']
    Uz = np.zeros_like(Ux)  # 2-D = 0

    df['rho_UxUx'] = rho * Ux * Ux
    df['rho_UxUy'] = rho * Ux * Uy
    df['rho_UxUz'] = np.zeros_like(Ux)

    df['rho_UyUy'] = rho * Uy * Uy
    df['rho_UyUz'] = np.zeros_like(Ux)

    df['rho_UzUz'] = np.zeros_like(Ux)

    # ============================================================
    # Prepack 2-D Sij and Rij
    # ============================================================
    Sij = {
        'xx': df['S_xx'], 'xy': df['S_xy'], 'xz': df['S_xz'],
        'yy': df['S_yy'], 'yz': df['S_yz'], 'zz': df['S_zz']
    }
    Rij = {
        'xy': df['R_xy'], 'xz': df['R_xz'], 'yz': df['R_yz']
    }

    # ============================================================
    # Viscous stress τᵢⱼ (2-D)
    # ============================================================
    mu = df['mu_suth']

    # divergence of velocity (2-D → NO dUz_dz)
    duk_dxk = df['dUx_dx'] + df['dUy_dy']
    scalar = (2 / 3) * duk_dxk

    df['Tao_xx'] = mu * (df['S_xx'] - scalar)
    df['Tao_yy'] = mu * (df['S_yy'] - scalar)
    df['Tao_zz'] = np.zeros_like(df['Tao_xx'])

    df['Tao_xy'] = mu * df['S_xy']
    df['Tao_yx'] = df['Tao_xy']

    df['Tao_xz'] = np.zeros_like(df['Tao_xx'])
    df['Tao_zx'] = np.zeros_like(df['Tao_xx'])
    df['Tao_yz'] = np.zeros_like(df['Tao_xx'])
    df['Tao_zy'] = np.zeros_like(df['Tao_xx'])

    # ============================================================
    # Advection ρ uⱼ ∂uᵢ/∂xⱼ (2-D)
    # ============================================================
    df['Adv_x'] = rho * (Ux * df['dUx_dx'] + Uy * df['dUx_dy'])
    df['Adv_y'] = rho * (Ux * df['dUy_dx'] + Uy * df['dUy_dy'])
    df['Adv_z'] = np.zeros_like(df['Adv_x'])

    # ============================================================
    # Strain advection ρ uⱼ Sᵢⱼ (2-D)
    # ============================================================
    df['strain_adv_x'] = rho * (Ux * Sij['xx'] + Uy * Sij['xy'])
    df['strain_adv_y'] = rho * (Ux * Sij['xy'] + Uy * Sij['yy'])
    df['strain_adv_z'] = np.zeros_like(df['strain_adv_x'])

    # ============================================================
    # Rotation advection ρ uⱼ Ωᵢⱼ (2-D)
    # ============================================================
    df['rot_adv_x'] = rho * (Uy * Rij['xy'])
    df['rot_adv_y'] = rho * (Ux * (-Rij['xy']))
    df['rot_adv_z'] = np.zeros_like(df['rot_adv_x'])

    # ============================================================
    # Magnitudes
    # ============================================================
    df['Strain_adv_mag'] = np.sqrt(df['strain_adv_x'] ** 2 + df['strain_adv_y'] ** 2)
    df['Rot_adv_mag'] = np.sqrt(df['rot_adv_x'] ** 2 + df['rot_adv_y'] ** 2)

    # ============================================================
    # Balance features
    # ============================================================
    df['S_vs_R'] = (df['Strain_mag'] - df['Rot_mag']) / (df['Strain_mag'] + df['Rot_mag'] + 1e-12)
    df['Str_pref_adv'] = df['Strain_adv_mag'] / (df['Strain_adv_mag'] + df['Rot_adv_mag'] + 1e-12)
    df['Rot_pref_adv'] = df['Rot_adv_mag'] / (df['Strain_adv_mag'] + df['Rot_adv_mag'] + 1e-12)

    # ============================================================
    # Cosine alignment (2-D)
    # ============================================================
    num = df['rot_adv_x'] * df['strain_adv_x'] + df['rot_adv_y'] * df['strain_adv_y']
    den = df['Rot_adv_mag'] * df['Strain_adv_mag'] + 1e-12
    df['cos_theta_adv'] = num / den
    # --- Pope/Ling tensor invariants (tr(S²), tr(R²), etc.) ---
    df = add_pope_invariants(df)

    # --- Velocity-gradient invariants (I1–I3, Q, R) ---
    df = add_velocity_gradient_invariants(df)
    return df


# ------------------------------------------------------------------------------
# === Tier 2: Post-Gradient Features ===========================================
# ------------------------------------------------------------------------------

def compute_rans_grad2_features(df):
    """
    2-D SAFE VERSION
    Computes second-order and tensor-derived quantities for a 2-D flow.
    Removes all z-terms and enforces planar divergence and momentum residuals.
    """

    # ============================================================
    # 1. Divergence of momentum flux: div(ρ U_i U_j)
    #    Valid derivatives for 2D:
    #       ∂(ρUxUx)/∂x + ∂(ρUxUy)/∂y
    #       ∂(ρUxUy)/∂x + ∂(ρUyUy)/∂y
    # ============================================================
    if all(c in df.columns for c in ['drho_UxUx_dx', 'drho_UxUy_dy']):
        df['div_rho_uiuj_x'] = df['drho_UxUx_dx'] + df['drho_UxUy_dy']
    else:
        df['div_rho_uiuj_x'] = 0.0

    if all(c in df.columns for c in ['drho_UxUy_dx', 'drho_UyUy_dy']):
        df['div_rho_uiuj_y'] = df['drho_UxUy_dx'] + df['drho_UyUy_dy']
    else:
        df['div_rho_uiuj_y'] = 0.0

    df['div_rho_uiuj_z'] = np.zeros_like(df['div_rho_uiuj_x'])   # always zero in 2D

    # ============================================================
    # 2. Compressibility vector: comp_i = div(ρUiUj) – Adv_i
    # ============================================================
    if 'Adv_x' in df.columns:
        df['comp_x'] = df['div_rho_uiuj_x'] - df['Adv_x']
        df['comp_y'] = df['div_rho_uiuj_y'] - df['Adv_y']
        df['comp_z'] = np.zeros_like(df['comp_x'])
    else:
        df['comp_x'] = df['comp_y'] = df['comp_z'] = 0.0

    # ============================================================
    # 3. Divergence of viscous stresses: div(τ)
    #    2D: ∂τxx/∂x + ∂τxy/∂y
    #        ∂τyx/∂x + ∂τyy/∂y
    # ============================================================
    if all(c in df.columns for c in ['dTao_xx_dx', 'dTao_xy_dy']):
        df['div_Tao_x'] = df['dTao_xx_dx'] + df['dTao_xy_dy']
    else:
        df['div_Tao_x'] = 0.0

    if all(c in df.columns for c in ['dTao_yx_dx', 'dTao_yy_dy']):
        df['div_Tao_y'] = df['dTao_yx_dx'] + df['dTao_yy_dy']
    else:
        df['div_Tao_y'] = 0.0

    df['div_Tao_z'] = np.zeros_like(df['div_Tao_x'])

    # ============================================================
    # 4. Momentum residuals: div(ρUiUj) + ∂p/∂xi − div(τ)
    # ============================================================
    # These pressure derivative names match your pipeline: dp_dx, dp_dy
    if all(c in df.columns for c in ['dp_dx', 'dp_dy']):
        df['resid_mom_x'] = df['div_rho_uiuj_x'] + df['dp_dx'] - df['div_Tao_x']
        df['resid_mom_y'] = df['div_rho_uiuj_y'] + df['dp_dy'] - df['div_Tao_y']
        df['resid_mom_z'] = np.zeros_like(df['resid_mom_x'])
    else:
        df['resid_mom_x'] = df['resid_mom_y'] = df['resid_mom_z'] = 0.0

    # ============================================================
    # 5. Add magnitudes (2D-safe version)
    # ============================================================
    df = add_tensor_and_vector_magnitudes(df)

    return df



# ------------------------------------------------------------------------------
# === Tensor/Invariant Helpers (embedded from your original) ===================
# ------------------------------------------------------------------------------
def tensor_from_df(df, prefix):
    """Return Nx3x3 tensor for 'S' or 'R' fields."""
    keys = [f'{prefix}_xx', f'{prefix}_xy', f'{prefix}_xz',
            f'{prefix}_yx', f'{prefix}_yy', f'{prefix}_yz',
            f'{prefix}_zx', f'{prefix}_zy', f'{prefix}_zz']
    M = np.zeros((len(df), 3, 3))
    M[:, 0, 0], M[:, 0, 1], M[:, 0, 2] = df[keys[0]], df[keys[1]], df[keys[2]]
    M[:, 1, 0], M[:, 1, 1], M[:, 1, 2] = df[keys[3]], df[keys[4]], df[keys[5]]
    M[:, 2, 0], M[:, 2, 1], M[:, 2, 2] = df[keys[6]], df[keys[7]], df[keys[8]]
    return M


def add_pope_invariants(df):
    """
    2-D safe version of Pope invariants.
    Computes invariants using 2×2 S and R, embeds into 3×3 with zeros.
    """

    # ===============================
    # Rebuild S and R as 2×2 matrices
    # ===============================
    S11 = df['S_xx'].values
    S12 = df['S_xy'].values
    S21 = df['S_yx'].values
    S22 = df['S_yy'].values

    R12 = df['R_xy'].values
    R21 = df['R_yx'].values

    # Build 2×2 S and R matrices for all points
    S2D = np.stack([
        np.stack([S11, S12], axis=1),
        np.stack([S21, S22], axis=1)
    ], axis=1)  # shape (N,2,2)

    R2D = np.stack([
        np.stack([np.zeros_like(R12), R12], axis=1),
        np.stack([R21, np.zeros_like(R12)], axis=1)
    ], axis=1)

    # ===============================
    # Tensor products in 2D
    # ===============================
    S2  = np.matmul(S2D, S2D)
    R2  = np.matmul(R2D, R2D)
    S3  = np.matmul(S2, S2D)
    S4  = np.matmul(S3, S2D)

    R2S  = np.matmul(R2, S2D)
    R2S2 = np.matmul(R2, S2)

    SR   = np.matmul(S2D, R2D)
    RS   = np.matmul(R2D, S2D)
    SRS  = np.matmul(SR, S2D)
    RSR  = np.matmul(RS, R2D)
    R4   = np.matmul(R2, R2)
    S2R2 = np.matmul(S2, R2)

    # ===============================
    # Trace operator in 2D
    # ===============================
    def tr2(A):
        return A[:, 0, 0] + A[:, 1, 1]

    df["tr_S2"]     = tr2(S2)
    df["tr_R2"]     = tr2(R2)
    df["tr_S3"]     = tr2(S3)
    df["tr_R2_S"]   = tr2(R2S)
    df["tr_R2_S2"]  = tr2(R2S2)
    df["tr_SR"]     = tr2(SR)
    df["tr_SRS"]    = tr2(SRS)
    df["tr_RSR"]    = tr2(RSR)
    df["tr_S4"]     = tr2(S4)
    df["tr_R4"]     = tr2(R4)
    df["tr_S2R2"]   = tr2(S2R2)

    # ===========================================================
    # Helper: embed 2×2 into full 3×3 (TBNN expects 9 components)
    # ===========================================================
    def embed_2D_to_3D(T):
        """
        Convert (N,2,2) → (N,3,3) with zeros in z components.
        """
        N = len(T)
        T3 = np.zeros((N, 3, 3))
        T3[:, 0:2, 0:2] = T
        return T3

    # ===============================
    # Embed all tensors back to 3D
    # ===============================
    tensors = {
        'S2':  S2,
        'R2':  R2,
        'S3':  S3,
        'R2S': R2S,
        'R2S2':R2S2,
        'SR':  SR,
        'RS':  RS,
        'SRS': SRS,
        'RSR': RSR,
        'S4':  S4,
        'R4':  R4,
        'S2R2':S2R2
    }

    # Store all tensors with 9-component labels
    for name, T2 in tensors.items():
        T3 = embed_2D_to_3D(T2)

        df = pd.concat([
            df,
            pd.DataFrame(
                {
                    f"{name}_xx": T3[:, 0, 0],
                    f"{name}_xy": T3[:, 0, 1],
                    f"{name}_xz": T3[:, 0, 2],
                    f"{name}_yx": T3[:, 1, 0],
                    f"{name}_yy": T3[:, 1, 1],
                    f"{name}_yz": T3[:, 1, 2],
                    f"{name}_zx": T3[:, 2, 0],
                    f"{name}_zy": T3[:, 2, 1],
                    f"{name}_zz": T3[:, 2, 2],
                },
                index=df.index
            )
        ], axis=1)

    return df


def add_velocity_gradient_invariants(df):
    """Add 2-D velocity-gradient invariants using A = [[dux/dx, dux/dy], [duy/dx, duy/dy]]."""

    # Build 2×2 gradient tensor A for each point
    A11 = df['dUx_dx']
    A12 = df['dUx_dy']
    A21 = df['dUy_dx']
    A22 = df['dUy_dy']

    # ---- Eigenvalues for each row ----
    # λ satisfy: λ^2 - (trA) λ + detA = 0
    trA = A11 + A22
    detA = A11 * A22 - A12 * A21

    disc = np.sqrt(np.maximum(trA**2 - 4*detA, 0.0))
    lambda1 = 0.5 * (trA + disc)
    lambda2 = 0.5 * (trA - disc)

    # ---- Classic invariants ----
    I1 = trA
    I2 = detA              # 2D analogue of second invariant
    # No third invariant in 2D
    Q = -I2                # consistent with TBNN convention
    R = np.zeros_like(I1)  # 2D velocity gradient has no cubic invariant
    # ---- Add to dataframe ----
    df = df.assign(
        lambda1_A=lambda1,
        lambda2_A=lambda2,
        I1_A=I1,
        I2_A=I2,
        Q_A=Q,
        R_A=R,
    )
    return df



def add_tensor_and_vector_magnitudes(df):
    """
    Compute magnitudes for key tensors/vectors (2-D safe).
    Only x,y-plane components are used.
    """

    # ==========================
    # TENSOR MAGNITUDES (2-D)
    # ==========================
    tensor_prefixes = [
        'S', 'R', 'S2', 'R2', 'S3', 'R2S', 'R2S2',
        'SR', 'RS', 'SRS', 'RSR', 'S4', 'R4', 'S2R2'
    ]

    # 2-D tensor components to include
    tensor_comps_2D = ['xx', 'xy', 'yx', 'yy']

    for prefix in tensor_prefixes:
        comps = [f'{prefix}_{c}' for c in tensor_comps_2D
                 if f'{prefix}_{c}' in df.columns]

        if comps:
            df[f'{prefix}_mag'] = np.sqrt(
                np.sum([(df[c]**2).values for c in comps], axis=0)
            )

    # ==========================
    # VECTOR MAGNITUDES (2-D)
    # ==========================
    vector_prefixes = [
        'Adv', 'strain_adv', 'rot_adv',
        'div_rho_uiuj', 'comp', 'div_Tao', 'resid_mom'
    ]

    vector_comps_2D = ['x', 'y']  # NO z

    for prefix in vector_prefixes:
        comps = [f'{prefix}_{c}' for c in vector_comps_2D
                 if f'{prefix}_{c}' in df.columns]

        if comps:
            df[f'{prefix}_mag'] = np.sqrt(
                np.sum([(df[c]**2).values for c in comps], axis=0)
            )

    return df
