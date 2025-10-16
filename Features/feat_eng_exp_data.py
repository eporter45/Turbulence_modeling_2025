# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 09:23:55 2025

@author: eoporter
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.tri as tri


#Dynamically resolve project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add to system path if not already
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

#print(PROJECT_ROOT)
from Data.Shear_mixing.boundary_conditions import BCs

idx = '1'
case = f'Case{idx}'

#data loading

def load_case_data_by_prefix(directory, prefix='Case1'):
    case_data = {}

    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith('.pkl'):
            fov_label = filename.split('_')[1]  # e.g., 'FOV1'
            file_path = os.path.join(directory, filename)
            try:
                df = pd.read_pickle(file_path)
                case_data[fov_label] = df
                print(f"✅ Loaded {filename} as '{fov_label}'")
            except Exception as e:
                print(f"❌ Failed to load {filename}: {e}")
    return case_data
'''         
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(project_root)
big_data_dir = os.path.join(project_root, 'data')
source_dir = os.path.join(big_data_dir, 'Shear_mixing')
exp_dir = os.path.join(source_dir, 'merged_exp')
print(exp_dir)
exp_dict = load_case_data_by_prefix(exp_dir, prefix=case)
'''

#Feature engineering

# Convert experimental coordinates to meters
'''
for df in exp_dict.values():
    df['Cx'] = df['x_mm'] / 1000.0
    df['Cy'] = df['y_mm'] / 1000.0
'''

def make_re_xxs(df, delta_U):
    df['Re_xx'] = df['uu'] / delta_U**2
    df['Re_xy'] = df['uv'] / delta_U**2
    df['Re_yy'] = df['vv'] / delta_U**2
    df['Re_zz'] = df['ww'] / delta_U**2
    # add other 2 items
    df['uw'] = np.zeros_like(df['uu'])
    df['vw'] = np.zeros_like(df['uu'])
    df['Re_xz'] = np.zeros_like(df['uu'])
    df['Re_yz'] = np.zeros_like(df['uu'])
    
    return df

def make_tke(df):
    df['tke'] =  0.5* (df['uu']+df['vv']+df['ww'])
    return df


def make_aij(df):
    k = df['tke']
    two_th_k = (2/3) * k
    df['a_xx'] = df['uu'] - two_th_k
    df['a_yy'] = df['vv'] - two_th_k
    df['a_zz'] = df['ww'] - two_th_k
    df['a_xy'] = df['uv']
    df['a_xz'] = df['uw']
    df['a_yz'] = df['vw']
    return df

def make_bij(df):
    k = df['tke']
    df['b_xx'] = (df['uu']/k) - (1/3) 
    df['b_yy'] =  (df['vv']/k) - (1/3) 
    df['b_zz'] =  (df['ww']/k) - (1/3) 
    df['b_xy'] = df['uv'] / k
    df['b_xz'] = df['uw'] / k
    df['b_yz'] = df['vw'] / k
    return df
    
def compute_invariants_aij(df):
    I_list = []
    II_list = []
    III_list = []
    eig_I_list = []
    eig_II_list = []
    eig_III_list = []
    

    for _, row in df.iterrows():
        A = np.array([
            [row['a_xx'], row['a_xy'], row['a_xz']],
            [row['a_xy'], row['a_yy'], row['a_yz']],
            [row['a_xz'], row['a_yz'], row['a_zz']]
        ])
        I = np.trace(A)
        II = np.trace(A @ A)
        III = np.linalg.det(A)
        
        eigs = np.linalg.eigvalsh(A)
        eigs = np.sort(eigs)[::-1]
        eig_I = np.sum(eigs)
        eig_II = np.sum(eigs**2)
        eig_III = np.prod(eigs)
        
        I_list.append(I)
        II_list.append(II)
        III_list.append(III)
        eig_I_list.append(eig_I)
        eig_II_list.append(eig_II)
        eig_III_list.append(eig_III)
        
    df['aij_I_comp'] = I_list    
    df['aij_II_comp'] = II_list
    df['aij_III_comp'] = III_list
    df['aij_I_eig'] = eig_I_list
    df['aij_II_eig'] = eig_II_list
    df['aij_III_eig'] =eig_III_list
    df['aij_eig1'] = eigs[0]
    df['aij_eig2'] = eigs[1]
    df['aij_eig3'] = eigs[2]
    return df    

def compute_invariants_bij(df):
    I_list = []
    II_list = []
    III_list = []
    eig_I_list = []
    eig_II_list = []
    eig_III_list = []

    for _, row in df.iterrows():
        B = np.array([
            [row['b_xx'], row['b_xy'], row['b_xz']],
            [row['b_xy'], row['b_yy'], row['b_yz']],
            [row['b_xz'], row['b_yz'], row['b_zz']]
        ])

        # Invariants from components
        I = np.trace(B)
        II = np.trace(B @ B)
        III = np.linalg.det(B)

        # Invariants from eigenvalues
        eigs = np.linalg.eigvalsh(B)  # For symmetric matrix
        eigs = np.sort(eigs)[::-1]
        eig_I = np.sum(eigs)
        eig_II = np.sum(eigs**2)
        eig_III = np.prod(eigs)

        # Append all
        I_list.append(I)
        II_list.append(II)
        III_list.append(III)
        eig_I_list.append(eig_I)
        eig_II_list.append(eig_II)
        eig_III_list.append(eig_III)

    df['bij_I_comp'] = I_list    
    df['bij_II_comp'] = II_list
    df['bij_III_comp'] = III_list
    df['bij_I_eig'] = eig_I_list
    df['bij_II_eig'] = eig_II_list
    df['bij_III_eig'] = eig_III_list
    df['bij_eig1'] = eigs[0]
    df['bij_eig2'] = eigs[1]
    df['bij_eig3'] = eigs[2]
    return df

def get_bayecentric_coords(df):
    eig1 = df['bij_eig1']
    eig2 = df['bij_eig2']
    eig3 = df['bij_eig3']
    eigs = np.array([eig1, eig2, eig3])
    eigs = np.sort(eigs)[::-1]
    l1, l2, l3 = eigs

    C1 = l1 - l2
    C2 = 2 * (l2 - l3)
    C3 = 3 * l3 + 1  # Ensures C1 + C2 + C3 = 1

    df['baye_C1'] = C1
    df['baye_C2'] = C2
    df['baye_C3'] = C3

    return df
    
def calculate_skew(df):
    denom_u = df['uu'] ** (3/2)
    denom_v = df['vv'] ** (3/2)
    df['Skew_u'] = df['uuu'] / denom_u
    df['Skew_v'] = df['vvv'] / denom_v
    #add in w/z components too
    return df

def calculate_transport_ratio(df):
    rms_u = np.sqrt(df['uu'])
    rms_v = np.sqrt(df['vv'])
    #rms_w = np.sqrt(df['ww'])
    df['TR_uuu'] = df['uuu'] / (df['uu'] * rms_u)
    df['TR_vuu'] = df['vuu'] / (df['uv'] * rms_u)
    df['TR_uvv'] = df['uvv'] / (df['uv'] * rms_v)
    df['TR_vvv'] = df['vvv'] / (df['vv'] * rms_v)
    df['TR_uww'] = df['uww'] / (df['ww'] * rms_u)
    df['TR_vww'] = df['vww'] / (df['ww']*rms_v)
    #calc TR_full
    numerator = df['uuu'] + df['vuu'] + df['uvv'] + df['vvv'] + df['uww'] + df['vww']
    #www assumed to be zero
    denom = df['tke'] ** (3/2)
    df['TR_ijk'] = numerator / denom
    return df
    
    
    return df
    
def process_and_save_all_features(exp_dict, save=False, save_dir='', delta_U=1.0):
    """
    Processes all feature engineering steps on each dataframe in exp_dict and saves the results.

    Parameters:
        exp_dict (dict): Dictionary of FOV label → DataFrame
        save_dir (str): Directory where processed files will be saved
        delta_U (float): Scaling factor for Reynolds stress terms
    """
    os.makedirs(save_dir, exist_ok=True)

    for label, df in exp_dict.items():
        print(f"🔄 Processing {label}...")

        try:
            # Feature engineering pipeline
            df = make_re_xxs(df, delta_U=delta_U)
            df = make_tke(df)
            df = make_aij(df)
            df = make_bij(df)
            df = compute_invariants_aij(df)
            df = compute_invariants_bij(df)
            df = get_bayecentric_coords(df)
            df = calculate_skew(df)
            df = calculate_transport_ratio(df)
            
            
            if save:
                if save_dir != '':
                    # Save processed dataframe
                    output_path = os.path.join(save_dir, f"processed_{case}_{label}.pkl")
                    df.to_pickle(output_path)
                    print(f"✅ Saved: {output_path}")
                else:
                    raise ValueError('Error No save dir passed in')
        except Exception as e:
            print(f"❌ Error processing {label}: {e}")
    print(df.columns.tolist())   
'''         
bc = BCs[case]
delta_u = bc['Reference']['delta_U']
save_dir = os.path.join(source_dir, 'processed_exp')
process_and_save_all_features(exp_dict, False, save_dir, delta_U=delta_u)
'''