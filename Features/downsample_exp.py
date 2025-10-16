# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 20:25:35 2025

@author: eoporter
"""


import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import pickle as pkl
from Features.make_featuresets import get_feature_set
#Dynamically resolve project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add to system path if not already
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Features.make_fs_kde_scratch import simple_kde_plot, staggered_kde_plot, multi_feature_kde_plot

#Plotting config and setup
'''
idx = '2'
case = f'Case{idx}'
grad_type = 'MLS' #mls or og 
dnd = 'dim'
fov='FOV1'
filename = f'{case}_{dnd}_{fov}.pkl'
save=True


#load RANS data
rans_dir = os.path.join(PROJECT_ROOT, 'data', 'Shear_mixing', 'RANS', 'training')
mls_file_dir = os.path.join(rans_dir, 'MLS')
mls_filepath = os.path.join(mls_file_dir, filename)
rans_df = pd.read_pickle(mls_filepath)
#load exp data
exp_dir = os.path.join(PROJECT_ROOT, 'Data', 'Shear_mixing','Exp_data', f'{dnd}_exp')
exp_filename = f'{dnd}_{case}_{fov}.pkl'
exp_df = pd.read_pickle(os.path.join(exp_dir,exp_filename))

exp_cx_range = (exp_df['Cx'].min(), exp_df['Cx'].max())
exp_cy_range = (exp_df['Cy'].min(), exp_df['Cy'].max())
rans_cx_range = (rans_df['Cx'].min(), rans_df['Cx'].max())
rans_cy_range = (rans_df['Cy'].min(), rans_df['Cy'].max())

print(f' RANS x range: {rans_cx_range}')
print(f' EXP x range: {exp_cx_range}')
print(f' RANS y range: {rans_cy_range}')
print(f' EXP y range: {exp_cy_range}')
'''
from scipy.spatial import cKDTree

def downsample_to_mesh(source_df, target_df, coord_cols=['Cx', 'Cy']):
    tree = cKDTree(source_df[coord_cols].values)
    _, indices = tree.query(target_df[coord_cols].values, k=1)
    return source_df.iloc[indices].reset_index(drop=True)

def run_downsample(cases, fovs, rans_path, exp_path, save=False, savepath=''):
    dnds = ['dim', 'nondim']  # Make sure your lists align or loop correctly
    from itertools import product
    for case, fov, dnd in product(cases, fovs, dnds):
        rans_filename = f'{case}_{dnd}_{fov}.pkl'
        exp_filename = f'{dnd}_{case}_{fov}.pkl'

        rans_file = os.path.join(rans_path, rans_filename)
        exp_file = os.path.join(exp_path, f'{dnd}_exp', exp_filename)

        rans_df = pd.read_pickle(rans_file)
        exp_df = pd.read_pickle(exp_file)

        downsampled = downsample_to_mesh(exp_df, rans_df)

        if save:
            if not savepath:
                raise ValueError('No save path entered')
            file_path = os.path.join(savepath, dnd)
            os.makedirs(file_path, exist_ok=True)
            saved_path = os.path.join(file_path,f'p_{exp_filename}')
            downsampled.to_pickle(saved_path)
            print(f'Saved {exp_filename} to {file_path}')
        print(f'{case} {fov} {dnd}')
        print(f'Length of EXP : {len(exp_df)}')
        print(f'Length of RANS: {len(rans_df)}')
        print(f'Length of Downsampled: {len(downsampled)}')
        print('Finished Downsampling')

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
shear_mixing_dir = os.path.join(PROJECT_ROOT, 'Data', 'Shear_mixing')
rans_dir = os.path.join(shear_mixing_dir, 'RANS', 'training', 'MLS')
exp_dir = os.path.join(shear_mixing_dir, 'EXP', 'Exp_data')
down_save_dir = os.path.join(shear_mixing_dir,'EXP', 'train_exp')
os.makedirs(down_save_dir, exist_ok=True)

# Config

cases = ['Case2']
idxs = ['1','2', '3', '4']
fovs = [f'FOV{idx}' for idx in idxs]
print(fovs)
run_downsample(cases, fovs, rans_dir, exp_dir, save=True, savepath=down_save_dir)