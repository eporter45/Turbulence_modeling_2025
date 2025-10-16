# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 11:22:46 2025

@author: eoporter
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
print(project_root)
import pandas as pd

idx = '2'
case = f'Case{idx}'
filename = f'CFD_{case}_Results.txt'
savedir = os.path.join(project_root, 'Shear_mixing','RANS', 'Clean')
case_dir = os.path.join(project_root, 'Shear_mixing','Raw_data', 'RANS')
def save_raw_rans_df(filename, file_dir, save_dir, save_name):
    file_path = os.path.join(case_dir, filename)
    # Read the file with pandas, specifying the columns and skipping the header line if necessary
    # You might need to skip the first line or set header=None depending on your file format
    df = pd.read_csv(file_path,header=0)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        'x-coordinate': 'Cx',
        'y-coordinate': 'Cy',
        'z-coordinate': 'Cz',
        'rexx': 'uu',
        'reyy': 'vv',
        'rezz': 'ww',
        'rexy': 'uv',
        'density': 'rho',
        'x-velocity': 'Ux',
        'y-velocity': 'Uy',
        'z-velocity': 'Uz',
        'turb-kinetic-energy': 'k',
        'specific-diss-rate': 'omega',
        'viscosity-turb': 'mu_t',
        'dx-velocity-dx': 'dUx_dx',
        'dy-velocity-dx': 'dUy_dx',
        'dz-velocity-dx': 'dUz_dx',
        'dx-velocity-dy': 'dUx_dy',
        'dy-velocity-dy': 'dUy_dy',
        'dz-velocity-dy': 'dUz_dy',
        'dx-velocity-dz': 'dUx_dz',
        'dy-velocity-dz': 'dUy_dz',
        'dz-velocity-dz': 'dUz_dz',
        'dp-dx': 'dp_dx',
        'dp-dy': 'dp_dy',
        'dp-dz': 'dp_dz',
        'mach': 'Mach',
        'pressure': 'p',
        'temperature': 'T',
        'cellnumber': 'cellID'  # Optional
    })
    
    # Optional: check dataframe head
    print(df.head())
    print(df.columns)   
    # Save as pickle
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    df.to_pickle(save_path)
    print(f'[INFO] saved {filename} as {save_name} in {save_path}')
    
    
save_name = f'{case}_pre_grads.pkl'
save_raw_rans_df(filename, case_dir, savedir, save_name)
