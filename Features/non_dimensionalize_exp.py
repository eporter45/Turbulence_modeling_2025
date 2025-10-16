# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 09:40:38 2025

@author: eoporter
"""

from Data.Shear_mixing.boundary_conditions import BCs


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
    

#config for running
idx = '4'
case = f'Case{idx}'
startswith = f'processed_{case}'
save=True


#load RANS data
exp_dir = os.path.join(PROJECT_ROOT, 'Data', 'Shear_mixing', 'processed_exp')
save_dir = os.path.join(PROJECT_ROOT, 'Data', 'Shear_mixing','nondim_exp')
os.makedirs(save_dir, exist_ok=True)


NON_DIM_KEYS = [
        'Re_xx', 'Re_xy', 'Re_yy', 'Re_zz', 'Re_xz', 'Re_yz',
        'a_xx', 'a_yy', 'a_zz', 'a_xy', 'a_xz', 'a_yz',
        'b_xx', 'b_yy', 'b_zz', 'b_xy', 'b_xz', 'b_yz',
        'aij_I_comp', 'aij_II_comp', 'aij_III_comp',
        'aij_I_eig', 'aij_II_eig', 'aij_III_eig',
        'aij_eig1', 'aij_eig2', 'aij_eig3',
        'bij_I_comp', 'bij_II_comp', 'bij_III_comp',
        'bij_I_eig', 'bij_II_eig', 'bij_III_eig',
        'bij_eig1', 'bij_eig2', 'bij_eig3',
        'baye_C1', 'baye_C2', 'baye_C3',
        'Skew_u', 'Skew_v',
        'TR_uuu', 'TR_vuu', 'TR_uvv', 'TR_vvv', 'TR_uww', 'TR_vww', 'TR_ijk'
    ]
SCALE_BY_L_REF = ['Cx', 'Cy']
SCALE_BY_U_REF = ['U', 'V', 'W', 'V_mag']
DIVIDE_BY_U2 = ['tke', 'uu', 'vv', 'ww', 'uv', 'uw', 'vw']
DIVIDE_BY_U3 = ['uuu', 'vvv', 'uvv', 'uww', 'vuu', 'vww']
DIVIDE_BY_U4 = ['uuuu', 'vvvv', 'wwww']

def get_exp_scaling_factor(col, BC):
    refs = BC['Reference']
    u_ref = refs['delta_U']
    l_ref = refs['x_ref']
    if col in NON_DIM_KEYS:
        return 1.0
    if col in SCALE_BY_L_REF:
        return 1 / l_ref
    if col in SCALE_BY_U_REF:
        return 1/ u_ref
    if col in DIVIDE_BY_U2:
        return 1/ (u_ref**2)
    if col in DIVIDE_BY_U3:
        return 1/ (u_ref**3)
    if col in DIVIDE_BY_U4:
        return 1/ (u_ref**4)
    #else:
        #print(f'[ERR-DEBUG] col: {col} not in groups, passing 1.0')
    return 1.0


def nondimensionalize_exp_df(df, bcs):
    df_nd = df.copy()
    for col in df.columns:
        factor = get_exp_scaling_factor(col, bcs)
        if factor != 1.0:
            df_nd[col] = df[col] * factor
    return df_nd






for file in os.listdir(exp_dir):
    if not file.startswith(startswith) or not file.endswith('.pkl'):
        continue
    try:
        case = file.split('_')[1]
        fov = file.split('_')[2].split('.')[0]
        bcs = BCs[case]
        df = pd.read_pickle(os.path.join(exp_dir, file))
        df_nd = nondimensionalize_exp_df(df, bcs)

        save_path = os.path.join(save_dir, f'nondim_{case}_{fov}.pkl')
        if save:
            df_nd.to_pickle(save_path)
            print(f"✅ Saved: {save_path}")
        else:
            print(f"✔️ Processed: {file} (not saved)")
    except Exception as e:
        print(f"❌ Error processing {file}: {e}")