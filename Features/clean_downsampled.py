# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 21:13:45 2025

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
idx = '2'
case = f'Case{idx}'
startswith = f'p_{case}'
save=True






#load downsampled exp data
shear_mixing_dir = os.path.join(PROJECT_ROOT, 'Data', 'Shear_mixing')
down_dir = os.path.join(shear_mixing_dir,'EXP', 'train_exp')
cases = ['Case1']
idxs = ['1', '2', '3', '4']
fovs = [f'FOV{idx}' for idx in idxs]
print(fovs)
dnds = ['dim', 'nondim']
dnd = dnds[1]
fov = fovs[1]

from Features.feat_eng_exp_data import (make_tke, make_aij, make_bij, 
                                        compute_invariants_aij,
                                        compute_invariants_bij, get_bayecentric_coords,
                                        calculate_skew, calculate_transport_ratio
                                        )


def recalc_df(df):
    df_out = make_tke(df)
    df_out = make_aij(df_out)
    df_out = make_bij(df_out)
    df_out = compute_invariants_aij(df_out)
    df_out = compute_invariants_bij(df_out)
    df_out = get_bayecentric_coords(df_out)
    df_out = calculate_skew(df_out)
    df_out = calculate_transport_ratio(df_out)
    return df_out


def run_recalcs(downsampled_dir, cases, fovs, save=False, savepath = ''):
    dnds = ['dim', 'nondim']
    from itertools import product
    for dnd, case, fov in product(dnds, cases, fovs):
        exp_filename = f'{dnd}_{case}_{fov}.pkl'
        exp_file = os.path.join(down_dir, dnd, exp_filename)
        exp = pd.read_pickle(exp_file)
        new_exp = recalc_df(exp)
        if save:
            if not savepath:
                raise ValueError('No save path entered')
            file_path = os.path.join(savepath, dnd)
            os.makedirs(file_path, exist_ok=True)
            saved_path = os.path.join(file_path, exp_filename)
            new_exp.to_pickle(saved_path)
            print(f'Saved {exp_filename} to {file_path}')
        print(f'Finished on {case} {fov} {dnd}')
    print('Finished recalcing')


save_dir = os.path.join(shear_mixing_dir, 'train_exp')
run_recalcs(down_dir, cases, fovs, save=True, savepath=save_dir)

