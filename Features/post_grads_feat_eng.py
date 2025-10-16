# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 08:59:32 2025

@author: eoporter
"""

import os
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import torch
import pandas as pd
import sys
from Shear_mixing.boundary_conditions import BCs


idx = '2'
case = f'Case{idx}'
gtype = 'MLS'
num = '3'
grads_type = f'{gtype}_grads'
filename = f'{case}_{gtype}_grad{num}.pkl'

first_second = 'second'
# Set project root and directory structure
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
print(project_root)
# Define key paths
big_data_dir = os.path.join(project_root, 'data')
source_dir = os.path.join(big_data_dir, 'Shear_mixing', 'RANS')
filepath = os.path.join(source_dir, grads_type ,first_second ,filename)
print(filepath)
df = pd.read_pickle(filepath)

print(df.columns.tolist())

rans_shift = {'Case1': 0.3628,
              'Case2': 0.3628, 
              'Case3': 0.0000,}


def add_comp_flux_div(df):
    #  d(rho_ui_uj)_dxj
    
    df['div_rho_uiuj_x'] = (df['drho_UxUx_dx'] +
                            df['drho_UxUy_dy'] +
                            df['drho_UxUz_dz'])
    df['div_rho_uiuj_y'] = (df['drho_UxUy_dx'] +
                            df['drho_UyUy_dy'] + 
                            df['drho_UyUz_dz'])
    df['div_rho_uiuj_z'] = (df['drho_UxUz_dx'] + 
                            df['drho_UyUz_dy'] + 
                            df['drho_UzUz_dz'])
    return df

def add_advection_vector(df):
    #advection is rho * uj *dui_dxj
    rho = df['rho']
    df['Adv_x'] = rho * (df['Ux'] * df['dUx_dx'] + 
                         df['Uy'] + df['dUx_dy'] + 
                         df['Uz'] * df['dUx_dz'])
    df['Adv_y'] = rho * (df['Ux'] * df['dUy_dx'] + 
                         df['Uy'] + df['dUy_dy'] + 
                         df['Uz'] * df['dUy_dz'])
    df['Adv_z'] = rho * (df['Ux'] * df['dUz_dx'] + 
                         df['Uy'] + df['dUz_dy'] + 
                         df['Uz'] * df['dUz_dz'])
    
    return df

def add_advection_tensor(df):
    #adv_ij = rho  * uj * dui_dxj
    rho = df['rho']
    df['Adv_xx'] = rho * df['Ux'] * df['dUx_dx']
    df['Adv_xy'] = rho * df['Uy'] * df['dUx_dy']
    df['Adv_xz'] = rho * df['Uz'] * df['dUx_dz'] 
    df['Adv_yx'] = rho * df['Ux'] * df['dUy_dx']    
    df['Adv_yy'] = rho * df['Uy'] * df['dUy_dy']    
    df['Adv_yz'] = rho * df['Uz'] * df['dUy_dz']
    df['Adv_zx'] = rho * df['Ux'] * df['dUz_dx']    
    df['Adv_zy'] = rho * df['Uy'] * df['dUz_dy']    
    df['Adv_zz'] = rho * df['Uz'] * df['dUz_dz']    

    return df


def add_compressibility_vector(df):
   df['comp_x'] = df['div_rho_uiuj_x'] - df['Adv_x']
   df['comp_y'] = df['div_rho_uiuj_y'] - df['Adv_y']
   df['comp_z'] = df['div_rho_uiuj_z'] - df['Adv_z']
   return df

def add_compressibility_tensor(df):
    df['comp_xx'] = df['drho_UxUx_dx'] - df['Adv_xx']
    df['comp_xy'] = df['drho_UxUy_dy'] - df['Adv_xy']
    df['comp_xz'] = df['drho_UxUz_dz'] - df['Adv_xz']
    
    df['comp_yx'] = df['drho_UxUy_dx'] - df['Adv_yx']
    df['comp_yy'] = df['drho_UyUy_dy'] - df['Adv_yy']
    df['comp_yz'] = df['drho_UyUz_dz'] - df['Adv_yz']
    
    df['comp_zx'] = df['drho_UxUz_dx'] - df['Adv_zx']
    df['comp_zy'] = df['drho_UyUz_dy'] - df['Adv_zy']
    df['comp_zz'] = df['drho_UzUz_dz'] - df['Adv_zz']
    
    return df

def add_advection_decomposed(df):
    rho = df['rho']
    Ux, Uy, Uz = df['Ux'], df['Uy'], df['Uz']

    # Strain tensor S_ij (symmetric)
    S_xx = df['dUx_dx']
    S_yy = df['dUy_dy']
    S_zz = df['dUz_dz']
    
    S_xy = 0.5 * (df['dUx_dy'] + df['dUy_dx'])
    S_yx = S_xy  # symmetry
    S_xz = 0.5 * (df['dUx_dz'] + df['dUz_dx'])
    S_zx = S_xz  # symmetry
    S_yz = 0.5 * (df['dUy_dz'] + df['dUz_dy'])
    S_zy = S_yz  # symmetry

    # Rotation tensor Omega_ij (antisymmetric)
    W_xy = 0.5 * (df['dUx_dy'] - df['dUy_dx'])
    W_yx = -W_xy
    W_xz = 0.5 * (df['dUx_dz'] - df['dUz_dx'])
    W_zx = -W_xz
    W_yz = 0.5 * (df['dUy_dz'] - df['dUz_dy'])
    W_zy = -W_yz

    # Strain convection: rho * u_j * S_ij
    strain_adv_x = rho * (Ux * S_xx + Uy * S_xy + Uz * S_xz)
    strain_adv_y = rho * (Ux * S_yx + Uy * S_yy + Uz * S_yz)
    strain_adv_z = rho * (Ux * S_zx + Uy * S_zy + Uz * S_zz)

    # Rotation convection: rho * u_j * Omega_ij
    rot_adv_x = rho * (Ux * 0 + Uy * W_xy + Uz * W_xz)      # Omega_xx = 0
    rot_adv_y = rho * (Ux * W_yx + Uy * 0 + Uz * W_yz)      # Omega_yy = 0
    rot_adv_z = rho * (Ux * W_zx + Uy * W_zy + Uz * 0)      # Omega_zz = 0

    # Add to dataframe
    df['strain_adv_x'] = strain_adv_x
    df['strain_adv_y'] = strain_adv_y
    df['strain_adv_z'] = strain_adv_z

    df['rot_adv_x'] = rot_adv_x
    df['rot_adv_y'] = rot_adv_y
    df['rot_adv_z'] = rot_adv_z

    return df

def add_viscous_stress_divergence_vector(df):
    # Assuming your df has derivatives of Tao_*_dx, Tao_*_dy, Tao_*_dz as derivatives of viscous stress tensor
   df['div_Tao_x'] = df['dTao_xx_dx'] + df['dTao_xy_dy'] + df['dTao_xz_dz']
   df['div_Tao_y'] = df['dTao_xy_dx'] + df['dTao_yy_dy'] + df['dTao_yz_dz']
   df['div_Tao_z'] = df['dTao_xz_dx'] + df['dTao_yz_dy'] + df['dTao_zz_dz']
   return df


def add_vicsous_tress_divergence_tensor(df):
    
    return df
def calculate_residuals(df):
    # -div(RST) = rho_uiuj_dxj -+dp_dxi = dTaoij_dxj
    
    df['resid_mom_x'] = df['div_rho_uiuj_x'] + df['dp_dx'] - df['div_Tao_x']
    df['resid_mom_y'] = df['div_rho_uiuj_y'] + df['dp_dy'] - df['div_Tao_y']
    df['resid_mom_z'] = df['div_rho_uiuj_z'] + df['dp_dz'] - df['div_Tao_z']
    return df


def compute_all_terms(df):
    df = add_comp_flux_div(df)
    df = add_advection_vector(df)
    df = add_advection_tensor(df)
    df = add_compressibility_vector(df)
    df = add_compressibility_tensor(df)
    df = add_advection_decomposed(df)
    df = add_viscous_stress_divergence_vector(df)
    df = add_vicsous_tress_divergence_tensor(df)  # currently empty but included for completeness
    df = calculate_residuals(df)
    return df

grads_type = f'{gtype}_grads'
first_second= 'all_feats'
savepath = os.path.join(source_dir, grads_type, first_second)
os.makedirs(savepath, exist_ok=True)
save_name = f'{case}_dim_all_feats.pkl'

df = compute_all_terms(df)
print(df.columns)
# Save separately
save_path = os.path.join(savepath, save_name)
print(f"Saving to: {save_path} (type: {type(save_path)})")

df.to_pickle(save_path)
print('✅ Save successful!')
# Save with overwrite protection

print(df.columns.tolist())