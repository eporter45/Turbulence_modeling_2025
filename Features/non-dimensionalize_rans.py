# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 09:40:38 2025

@author: eoporter
"""



import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import pickle as pkl
#Dynamically resolve project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add to system path if not already
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
from Data.Shear_mixing.boundary_conditions import BCs


#config for running
idx = '2'
case = f'Case{idx}'
gtype = 'MLS'
grad_type = f'{gtype}_grads'
#grad_type = 'og_grads'
filename = f'{case}_dim_all_feats.pkl'
save_name = f'{case}_nondim_all_feats.pkl'
save=True


#load RANS data
rans_dir = os.path.join(PROJECT_ROOT, 'data', 'Shear_mixing', 'RANS')
file_dir = os.path.join(rans_dir, grad_type, 'all_feats')
filepath = os.path.join(file_dir, filename)
savepath = os.path.join(file_dir, save_name)
df = pd.read_pickle(filepath)
print(df.columns.tolist())





NO_SCALING = [ 'cellID','Q_norm_S']
SCALE_BY_L_REF = ['Cx', 'Cy', 'Cz']
SCALE_BY_U_REF = ['Ux', 'Uy', 'Uz']
DIVIDE_BY_U2 = ['k', 'uu', 'vv', 'ww', 'uv']
DIVIDE_BY_Mu = ['mu_suth']
def get_scaling_factor(feature_key, bcs):
    # Untouched features
    scalars = bcs['Reference']
    l_ref = scalars['x_ref']
    u_ref = scalars['delta_U']
    mu_ref = scalars['mu_ref']
    p_ref = scalars['P_ref'] * 1000 #kpa --> pa
    T_ref = scalars['T_ref']
    rho_ref = scalars['rho_ref']
    re_ref = (rho_ref * u_ref * l_ref ) / mu_ref
    if feature_key in NO_SCALING:
        return 1.0
    
    if feature_key in DIVIDE_BY_Mu:
        return 1 / mu_ref
    # Coordinates
    if feature_key in SCALE_BY_L_REF:
        return 1 / l_ref
    
    # Velocity
    if feature_key in SCALE_BY_U_REF:
        return 1 / u_ref
    
    # Decomposed advection terms (strain and rotation advection)
    if feature_key.startswith('strain_adv_') or feature_key.startswith('rot_adv_'):
        return l_ref / u_ref

    # Kinetic energy / Reynolds stresses
    if feature_key in DIVIDE_BY_U2:
        return l_ref / (rho_ref * u_ref**2)
    if feature_key == 'T':
        return 1/T_ref
    if feature_key.startswith('dT_') and '_d' in feature_key:
        return l_ref/T_ref
    
    # Velocity gradients (dU/dx)
    if feature_key.startswith('dU') and '_d' in feature_key:
        # scale by L_ref / U_ref
        return l_ref/u_ref
    
    # Double velocity derivatives (ddU/dxdy)
    if feature_key.startswith('ddU'):
        return (l_ref**2) / u_ref
    
    # Strain or rotation tensors S_, R_
    if feature_key.startswith('S_') or feature_key.startswith('R_'):
        return l_ref / u_ref
    
    # Pressure
    if feature_key == 'p':
        #pref is stored in kPa
        return 1 / p_ref
    
    # Pressure gradients dp_dx, etc
    if feature_key.startswith('dp_d'):
        return l_ref / p_ref
    
    # Advection and compressibility terms
    if (
        ('Adv' in feature_key and not feature_key.startswith('strain_adv_') and not feature_key.startswith('rot_adv_')) or
        'rho_U' in feature_key or
        feature_key.startswith('drho_') or
        feature_key.startswith('comp_')
    ):
        return l_ref / (rho_ref * u_ref**2)
    
    # Tao tensor terms (viscous stress)
    if feature_key.startswith('Tao_') and not feature_key.startswith('dTao_'):
        return l_ref / re_ref
    
    # Gradient of Tao tensor (dTao)
    if feature_key.startswith('dTao_') or feature_key.startswith('div_Tao'):
        return 1.0 / re_ref    # gradients/divergence scale the same way
    
    # Viscous divergence and residuals
    if feature_key.startswith('resid_mom'):
        return l_ref / (rho_ref * (u_ref **2))
    
    # Default fallback
    return 1.0


def nondimensionalize_df(df, bcs):
    """
    Non-dimensionalize features in DataFrame df using scaling factors from bcs.

    """
    df_nd = df
    for col in df.columns:
        scale = get_scaling_factor(col, bcs)
        if scale != 1.0:
            df_nd[col] = df[col] * scale
    return df_nd


# make nondim df
bcs = BCs[case]
feats = ['uv', 'Ux', 'dUx_dx', 'p', 'Adv_y', 'rho', 'k', 'mu_suth', 'dp_dy']


for feat in feats:
    print(df[feat].describe())
df_nd = nondimensionalize_df(df, bcs)
for feat in feats:
    print(df_nd[feat].describe())
if save:
    df_nd.to_pickle(savepath)
    print(f'Saved to {savepath}')