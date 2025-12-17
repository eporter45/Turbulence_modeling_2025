# ===============================================================
# non_dimensionalize.py
# Unified non-dimensionalization for RANS and EXP data
# ===============================================================

import numpy as np
import pandas as pd
from Data.Shear_mixing.boundary_conditions import BCs
# === RANS + EXP Non-Dimensionalization Scaling ==============================
# Quantities that remain nondimensional or excluded from scaling
NO_SCALING = [
    'cellID',
    'b_xx','b_yy','b_zz','b_xy','b_xz','b_yz',
    'b_I','b_II','b_III','b_eig1','b_eig2','b_eig3',
    'S_vs_R','Q_norm', 'Re'
]
# Base coordinate & primitive velocity scaling
SCALE_BY_L_REF = ['Cx','Cy','Cz']
SCALE_BY_U_REF = ['Ux','Uy','Uz','U','V','W','V_mag']
# Standard turbulence quantities (scalar energy-like)
DIVIDE_BY_U2 = ['k','tke','uu','vv','ww','uv','uw','vw',
                'a_xx','a_yy','a_zz','a_xy','a_xz','a_yz',
                'a_I','a_II','a_III','a_eig1','a_eig2','a_eig3']
DIVIDE_BY_U3 = ['uuu','vvv','uvv','uww','vuu','vww']
DIVIDE_BY_U4 = ['uuuu','vvvv','wwww']
DIVIDE_BY_MU = ['mu_suth']
# --- (L_ref / U_ref)^n scaling for gradient, tensor, and invariant terms ----
SCALE_BY_LU = {
    1: [  # quantities ∝ (U/L)
        # Strain and rotation tensors
        'S_mag',
        'S_xx','S_xy','S_xz','S_yx','S_yy','S_yz','S_zx','S_zy','S_zz',
        'R_mag',
        'R_xx','R_xy','R_xz','R_yx','R_yy','R_yz','R_zx','R_zy','R_zz',
    ],

    2: [  # quantities ∝ (U/L)^2
        'tr_S2','tr_R2','tr_SR',
        # Pope tensors: quadratic
        'S2_mag','R2_mag','SR_mag','RS_mag',
        'S2_xx','S2_xy','S2_xz','S2_yx','S2_yy','S2_yz','S2_zx','S2_zy','S2_zz',
        'R2_xx','R2_xy','R2_xz','R2_yx','R2_yy','R2_yz','R2_zx','R2_zy','R2_zz',
        'SR_xx','SR_xy','SR_xz','SR_yx','SR_yy','SR_yz','SR_zx','SR_zy','SR_zz',
        'RS_xx','RS_xy','RS_xz','RS_yx','RS_yy','RS_yz','RS_zx','RS_zy','RS_zz',
    ],

    3: [  # quantities ∝ (U/L)^3
        'tr_S3','tr_R2_S','tr_SRS','tr_RSR',
        # Pope tensors: cubic
        'S3_mag','R2S_mag','SRS_mag','RSR_mag',
        'S3_xx','S3_xy','S3_xz','S3_yx','S3_yy','S3_yz','S3_zx','S3_zy','S3_zz',
        'R2S_xx','R2S_xy','R2S_xz','R2S_yx','R2S_yy','R2S_yz','R2S_zx','R2S_zy','R2S_zz',
        'SRS_xx','SRS_xy','SRS_xz','SRS_yx','SRS_yy','SRS_yz','SRS_zx','SRS_zy','SRS_zz',
        'RSR_xx','RSR_xy','RSR_xz','RSR_yx','RSR_yy','RSR_yz','RSR_zx','RSR_zy','RSR_zz',
    ],

    4: [  # quantities ∝ (U/L)^4
        'tr_S4','tr_R4','tr_S2R2',
        # Pope tensors: quartic
        'S4_mag','R4_mag','S2R2_mag',
        'S4_xx','S4_xy','S4_xz','S4_yx','S4_yy','S4_yz','S4_zx','S4_zy','S4_zz',
        'R4_xx','R4_xy','R4_xz','R4_yx','R4_yy','R4_yz','R4_zx','R4_zy','R4_zz',
        'S2R2_xx','S2R2_xy','S2R2_xz','S2R2_yx','S2R2_yy','S2R2_yz','S2R2_zx','S2R2_zy','S2R2_zz',
    ]   }



def get_rans_scaling_factor(feature_key, bcs):
    """
    Compute non-dimensional scaling factor for a given RANS feature.
    """
    ref = bcs['Reference']
    l_ref = ref['x_ref']
    u_ref = ref['delta_U']
    mu_ref = ref['mu_ref']
    p_ref = ref['P_ref'] * 1000  # kPa → Pa
    T_ref = ref['T_ref']
    rho_ref = ref['rho_ref']
    re_ref = (rho_ref * u_ref * l_ref) / mu_ref

    # --- Direct mappings ---
    if feature_key in NO_SCALING:
        return 1.0
    if feature_key in DIVIDE_BY_MU:
        return 1 / mu_ref
    if feature_key in SCALE_BY_L_REF:
        return 1 / l_ref
    if feature_key in SCALE_BY_U_REF:
        return 1 / u_ref
    if feature_key in DIVIDE_BY_U2:
        return 1 / (u_ref ** 2)
    if feature_key in DIVIDE_BY_U3:
        return 1 / (u_ref ** 3)
    if feature_key in DIVIDE_BY_U4:
        return 1 / (u_ref ** 4)

    # --- Thermodynamic & primitive gradients ---
    if feature_key == 'T':
        return 1 / T_ref
    if feature_key.startswith('dT_'):
        return l_ref / T_ref
    if feature_key == 'p':
        return 1 / p_ref
    if feature_key.startswith('dp_d'):
        return l_ref / p_ref

    # --- Transport / compressibility / advection / residuals ---
    if ('Adv' in feature_key and not feature_key.startswith(('strain_adv_', 'rot_adv_'))) \
            or 'rho_U' in feature_key or feature_key.startswith(('drho_', 'comp_')):
        return l_ref / (rho_ref * u_ref ** 2)
    if feature_key.startswith('Tao_') and not feature_key.startswith('dTao_'):
        return l_ref / re_ref
    if feature_key.startswith(('dTao_', 'div_Tao')):
        return 1 / re_ref
    if feature_key.startswith('resid_mom'):
        return l_ref / (rho_ref * u_ref ** 2)

    # --- Pope tensors & invariants scaling (L_ref / U_ref)^n ---
    for n, keys in SCALE_BY_LU.items():
        if feature_key in keys:
            return (l_ref / u_ref) ** n
    # --- Default ---
    return 1.0



def nondim_rans_df(df, case_name):
    """
    Apply non-dimensionalization to RANS DataFrame based on case BCs.
    """
    if case_name not in BCs:
        raise KeyError(f"[ERROR] Case '{case_name}' not found in BCs.")
    bcs = BCs[case_name]
    df_nd = df.copy()
    for col in df.columns:
        scale = get_rans_scaling_factor(col, bcs)
        if scale != 1.0:
            df_nd[col] = df[col] * scale
    return df_nd





def get_exp_scaling_factor(feature_key, bcs):
    """
    Compute non-dimensional scaling factor for experimental features.
    """
    ref = bcs['Reference']
    l_ref = ref['x_ref']
    u_ref = ref['delta_U']

    # --- Direct mappings ---
    if feature_key in NO_SCALING:
        return 1.0
    if feature_key in SCALE_BY_L_REF:
        return 1 / l_ref
    if feature_key in SCALE_BY_U_REF:
        return 1 / u_ref
    if feature_key in DIVIDE_BY_U2:
        return 1 / (u_ref ** 2)
    if feature_key in DIVIDE_BY_U3:
        return 1 / (u_ref ** 3)
    if feature_key in DIVIDE_BY_U4:
        return 1 / (u_ref ** 4)

    # --- Pope tensors & invariants scaling (L_ref / U_ref)^n ---
    for n, keys in SCALE_BY_LU.items():
        if feature_key in keys:
            return (l_ref / u_ref) ** n

    # --- Default ---
    return 1.0


def nondim_exp_df(df, case_name):
    """
    Apply non-dimensionalization to EXP DataFrame based on case BCs.
    """
    if case_name not in BCs.keys():
        raise KeyError(f"[ERROR] Case '{case_name}' not found in BCs.")
    bcs = BCs[case_name]
    df_nd = df.copy()
    for col in df.columns:
        scale = get_exp_scaling_factor(col, bcs)
        if scale != 1.0:
            df_nd[col] = df[col] * scale
    return df_nd


# ---------------------------------------------------------------
# === 3. PIPELINE ENTRY ==========================================
# ---------------------------------------------------------------
def apply_nondimensionalization(df, case_name, mode='rans'):
    """
    Wrapper for pipeline usage.

    Parameters
    ----------
    df : pd.DataFrame
        Data to be non-dimensionalized.
    case_name : str
        Case identifier.
    mode : str
        'rans' or 'exp'.
    """
    name = case_name.split('_')[0]
    if mode.lower() == 'rans':
        return nondim_rans_df(df, name)
    elif mode.lower() == 'exp':
        return nondim_exp_df(df, name)
    else:
        raise ValueError(f"[ERROR] Invalid nondim mode, {mode}, (must be 'rans' or 'exp').")