# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 11:42:40 2025
Uncomment idx to use
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
    
#config
idx = '2'
case = f'Case{idx}'
#rans_datasets to slice
gtype = 'MLS' #og or mls
grad_type = f'{gtype}_grads'
dnd = 'nondim' #dim or nondim
#load RANS data
rans_data_dir = os.path.join(PROJECT_ROOT, 'data', 'Shear_mixing', 'RANS')
rans_file_dir = os.path.join(rans_data_dir, 'training', gtype)
rans_filepath = os.path.join(rans_data_dir, grad_type, 'all_feats' ,f'{case}_{dnd}_all_feats.pkl' )
rans_df = pd.read_pickle(rans_filepath)
#print(f'[DEBUG] RANS filepath: \n {rans_filepath})
#print(f'[DEBUG] RANS df head: \n {rans_df.head})

#Load_exp_data
exp_data_dir = os.path.join(PROJECT_ROOT, 'data', 'Shear_mixing','EXP', 'Exp_data')
exp_file = os.path.join(exp_data_dir,f'{dnd}_exp')
exp_strt_w = f'{dnd}_{case}'
fovs = ['FOV1', 'FOV2', 'FOV3', 'FOV4']
slice_bounds = {}
for fov in fovs:
    try:
        path= os.path.join(exp_file, f'{exp_strt_w}_{fov}.pkl')
        df =pd.read_pickle(path)
        slice_bounds[fov] = {'x': (df['Cx'].min(), df['Cx'].max()),
                     'y': (df['Cy'].min(), df['Cy'].max())}
    finally:
        print(f'[INFO] {case} does not have {fov}')


rans_fovs = {}
df = rans_df
for key in slice_bounds.keys():
    bds = slice_bounds[key]
    subset_df = df[(df['Cx'] >= bds['x'][0]) &
                   (df['Cx'] <= bds['x'][1]) & 
                   (df['Cy'] >= bds['y'][0]) & 
                   (df['Cy'] <= bds['y'][1])]
    rans_fovs[key] = subset_df

for fov_key, sliced_df in rans_fovs.items():
    save_name = f"{case}_{dnd}_{fov_key}.pkl"
    save_path = os.path.join(rans_file_dir, save_name)
    sliced_df.to_pickle(save_path)
    print(f"[SAVED] {fov_key} slice saved to: {save_path}")
    


