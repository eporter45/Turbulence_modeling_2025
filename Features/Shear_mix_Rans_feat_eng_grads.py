# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 13:52:40 2025

@author: eoporter
"""

import os
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import torch
import pandas as pd
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
print(project_root)
from Data.Shear_mixing.boundary_conditions import BCs
# Set project root and directory structure


idx = '2'
case = f'Case{idx}'
gtype = 'MLS'

filename = f'{case}_{gtype}_grad1.pkl'
grads_type = f'{gtype}_grads'
first_second = 'first'
# Define key paths
big_data_dir = os.path.join(project_root, 'Data')
source_dir = os.path.join(big_data_dir, 'Shear_mixing', 'RANS')
load_dir = os.path.join(source_dir, 'Clean')
ddir = os.path.join(source_dir, grads_type, first_second)
load_path = os.path.join(ddir, filename)

#filepath = os.path.join(source_dir, grads_type ,first_second ,filename)
df = pd.read_pickle(load_path)

print(df.columns)

rans_shift = {'Case1': 0.3628,
              'Case2': 0.3628, 
              'Case3': 0.0000,}


def calculate_rho(df, R=287.05):
    t = df['T']
    p = df['p']
    return p / (R * t)

def calculate_mu(df, bcs):
    mu_ref = bcs['Reference']['mu_ref']
    T_ref = bcs['Reference']['T_ref']
    T = df['T']
    S = 110.4
    return mu_ref * (T / T_ref)**1.5 * (T_ref + S) / (T + S)

def add_Strain_tensor(df):
    df['S_xx'] = 0.5*(df['dUx_dx'] + df['dUx_dx'])
    df['S_xy'] = 0.5*(df['dUx_dy'] + df['dUy_dx'])
    df['S_yx'] = 0.5*(df['dUy_dx'] + df['dUx_dy'])
    df['S_yy'] = 0.5*(df['dUy_dy'] + df['dUy_dy'])
    df['S_xz'] = 0.5*(df['dUx_dz'] + df['dUz_dx'])
    df['S_zx'] = 0.5*(df['dUz_dx'] + df['dUx_dz'])
    df['S_yz'] = 0.5*(df['dUy_dz'] + df['dUz_dy'])
    df['S_zy'] = 0.5*(df['dUz_dy'] + df['dUy_dz'])
    df['S_zz'] = 0.5*(df['dUz_dz'] + df['dUz_dz'])
    return df

def add_rotation_tensor(df):
    df['R_xx'] = 0.5*(df['dUx_dx'] - df['dUx_dx'])
    df['R_xy'] = 0.5*(df['dUx_dy'] - df['dUy_dx'])
    df['R_yx'] = 0.5*(df['dUy_dx'] - df['dUx_dy'])
    df['R_yy'] = 0.5*(df['dUy_dy'] - df['dUy_dy'])
    df['R_xz'] = 0.5*(df['dUx_dz'] - df['dUz_dx'])
    df['R_zx'] = 0.5*(df['dUz_dx'] - df['dUx_dz'])
    df['R_yz'] = 0.5*(df['dUy_dz'] - df['dUz_dy'])
    df['R_zy'] = 0.5*(df['dUz_dy'] - df['dUy_dz'])
    df['R_zz'] = 0.5*(df['dUz_dz'] - df['dUz_dz'])
    return df

def add_Q_criterion(df):
    # Frobenius norms squared
    S_sq = (
        df['S_xx']**2 + df['S_yy']**2 + df['S_zz']**2 +
        2 * (df['S_xy']**2 + df['S_xz']**2 + df['S_yz']**2)
    )
    
    R_sq = (
        df['R_xy']**2 + df['R_xz']**2 + df['R_yz']**2
    ) * 2  # Only off-diagonal since R_ii = 0 always

    df['Q_crit'] = 0.5 * (R_sq - S_sq)
    df['Q_norm_S'] = df['Q_crit'] / (S_sq + 1e-12)  # Add small number to avoid divide-by-zero
    return df

def add_viscous_tensor(df):
    mu = df['mu_suth']
    duk_dxk = df['dUx_dx'] + df['dUy_dy'] + df['dUz_dz']
    df['dUi_dxi'] = duk_dxk
    scalar = (2*duk_dxk)/3
    df['Tao_xx'] = mu * (df['S_xx'] - scalar)
    df['Tao_yy'] = mu * (df['S_yy'] - scalar)
    df['Tao_zz'] = mu * (df['S_zz'] - scalar)
    df['Tao_xy'] = mu * df['S_xy']
    df['Tao_yx'] = mu * df['S_yx']
    df['Tao_xz'] = mu * df['S_xz']
    df['Tao_zx'] = mu * df['S_zx']
    df['Tao_yz'] = mu * df['S_yz']
    df['Tao_zy'] = mu * df['S_zy']
    return df
    
def add_advection_terms(df):
    #these are the advection terms d(rho_ui_uj)_dxj pre grads
    rho = df['rho']
    ux = df['Ux']
    uy = df['Uy']
    uz = df['Uz']
    df['rho_UxUx'] = rho * ux * ux
    df['rho_UxUy'] = rho * ux * uy
    df['rho_UxUz'] = rho * ux *uz
    df['rho_UyUy'] = rho * uy * uy
    df['rho_UyUz'] = rho * uy * uz
    df['rho_UzUz'] = rho * uz * uz
    return df

def shift_x_bounds(df, shifted_val):
    df['Cx'] = df['Cx'] - shifted_val
    return df


def add_non_grad_feats(df, bcs, case):
    bc = bcs[case]
    df_out = df.copy()
    df_out['rho'] = calculate_rho(df_out)
    df_out['mu_suth'] = calculate_mu(df_out, bc)
    df_out = add_Strain_tensor(df_out)
    df_out = add_rotation_tensor(df_out)
    df_out = add_viscous_tensor(df_out)
    df_out = add_advection_terms(df_out)
    #df_out = shift_x_bounds(df_out, rans_shift[case])
    df_out = add_Q_criterion(df_out)
    return df_out






import os
import pandas as pd
from datetime import datetime
gtype = 'MLS'
grads_type = f'{gtype}_grads'
first_second = 'second'
num = '1'
if first_second == 'first':
    num = '1'
else:
    num = '2'
save_name = f'{case}_{gtype}_grad2.pkl'
savepath = os.path.join(source_dir, grads_type, first_second)
os.makedirs(savepath, exist_ok=True)

df = add_non_grad_feats(df, BCs, case)
print(df.columns)
bds_cx = (df['Cx'].min(), df['Cx'].max())
ln_cx = len(df['Cx'])
print(f'length of cx and bounds:{bds_cx}, {ln_cx} ')
# Save separately
save_path = os.path.join(savepath, save_name)
print(f"Saving to: {save_path} (type: {type(save_path)})")
df.to_pickle(save_path)
print('✅ Save successful!')
# Save with overwrite protection

