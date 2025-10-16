# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 11:42:56 2025

@author: eoporter
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pickle as pkl
import numpy as np
import torch
import numpy as np
import pandas as pd
#for later stages, import the boundary conditions for ref length
filepath = "RANS/Results.txt"
# Construct the path relative to your script location
file_path = os.path.join(os.path.dirname(__file__), 'RANS', 'Results.txt')

# Load the data
df = pd.read_csv(file_path)
print(df.columns)
C_ax = 0.130023                                 # Axial chord [m]


# load data
cx = df['    x-coordinate'].values
cy = df['    y-coordinate'].values
cz = df['    z-coordinate'].values
ux = df['      x-velocity'].values
uy = df['      y-velocity'].values
pr = df['        pressure'].values
temp = df['     temperature'].values
#below are derrived using a subroutine when solving the k-omega sst eq's
rho_uv = df['       neg_rhouv']
rho_uu = df['       neg_rhouu']
rho_vv = df['       neg_rhovv']
rho_ut = df['       neg_rhout']
rho_vt = df['       neg_rhovt']

#start slicing
x_min, x_max = 1.0* C_ax, 1.5 *C_ax
# Apply mask to select data within that x-range
mask = (cx >= x_min) & (cx <= x_max)
df_sliced = df[mask].copy()

# First, strip leading/trailing whitespace from all column names
df_sliced.columns = df_sliced.columns.str.strip()
df_sliced = df_sliced.rename(columns={
    'x-coordinate': 'Cx',
    'y-coordinate': 'Cy',
    'z-coordinate': 'Cz',
    'x-velocity': 'Ux',
    'y-velocity': 'Uy',
    'pressure': 'p',
    'temperature': 'T',
    'neg_rhouu': 'rho_uu',  
    'neg_rhouv': 'rho_uv',       # Reynolds stress tensor
    'neg_rhovv': 'rho_vv',       # Reynolds stress tensor
    'neg_rhout': 'rho_uT',           # Turbulent heat flux
    'neg_rhovt': 'rho_vT'           # Turbulent heat flux

})
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Define filename and full save path under the project root
filename = "TE_sliced_data_1ax1p5.pkl"
save_dir = os.path.join(PROJECT_ROOT, 'Data', 'RANS')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, filename)

# Save the DataFrame
df_sliced.to_pickle(save_path)
print(f"[INFO] Saved sliced DataFrame to: {save_path}")


def scale_df_bounds(df, debug=False, save=False, save_path=''):
    required_cols = {'Cx', 'Cy'}
    # Check 1: Ensure required coordinate columns are present
    if not required_cols.issubset(df.columns):
        raise ValueError(f"[Error] DataFrame must contain columns: {required_cols}")
    cx = df['Cx'].values
    cy = df['Cy'].values
    # Check 2: Prevent redundant shifting
    if np.isclose(cx.min(), 0.0) or np.isclose(cy.min(), 0.0):
        raise ValueError("[Error] Cx or Cy already appears to be shifted (min is near zero).")
    # Check 3: Prevent overwriting original coordinate backup
    if 'Cx_original' in df.columns or 'Cy_original' in df.columns:
        raise ValueError("[Error] 'Cx_original' or 'Cy_original' already exists in DataFrame.")
   
    # Compute original bounds
    min_x = cx.min()
    min_y = cy.min()
    if debug:
        print(f"[Debug] Original X domain: [{min_x}, {cx.max()}]")
        print(f"[Debug] Original Y domain: [{min_y}, {cy.max()}]")
    # Save originals
    df['Cx_original'] = cx
    df['Cy_original'] = cy
    # Shift coordinates
    df['Cx'] = cx - min_x
    df['Cy'] = cy - min_y

    if debug:
        print(f"[Debug] Shifted X domain: [{df['Cx'].min()}, {df['Cx'].max()}]")
        print(f"[Debug] Shifted Y domain: [{df['Cy'].min()}, {df['Cy'].max()}]")

    if save:
        df.to_pickle(save_path)
        if debug:
            print(f"[Debug] Saved shifted DataFrame to: {save_path}")

    return df

df = scale_df_bounds(df_sliced, True, True, save_path)
import matplotlib.pyplot as plt

plt.scatter(df['Cx'], df['Cy'])
plt.show()
print(f"[INFO] Saved shifted bounds DataFrame to: {save_path}")




