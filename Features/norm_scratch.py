# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 18:02:00 2025

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
sets =['FS1', 'FS1_simple', 'FS2', 'FS3a',
       'FS3b', 'FS3c', 'FS4', 'FS5',
       'FS6', 'FS7']
sets = ['FS8']
#config for running
idx = '1'
case = f'Case{idx}'
grad_type = 'MLS' #mls or og 
d_nd = 'nondim' #dim or nondim
fov='FOV1'
filename = f'{case}_{d_nd}_{fov}.pkl'
save=True


#load RANS data
rans_dir = os.path.join(PROJECT_ROOT, 'data', 'Shear_mixing', 'RANS', 'training')
mls_file_dir = os.path.join(rans_dir, 'MLS')
og_file_dir = os.path.join(rans_dir, 'og')
nd_mls_filepath = os.path.join(mls_file_dir, filename)
nd_og_filepath = os.path.join(og_file_dir, filename)
nd_mls_filepath = os.path.join(mls_file_dir, filename)
nd_og_filepath = os.path.join(og_file_dir, filename)
nd_mls_df = pd.read_pickle(nd_mls_filepath)
nd_og_df = pd.read_pickle(nd_og_filepath)
d_nd = 'dim' #dim or nondim
filename = f'{case}_{d_nd}_{fov}.pkl'
dim_mls_filepath = os.path.join(mls_file_dir, filename)
dim_og_filepath = os.path.join(og_file_dir, filename)
dim_mls_df = pd.read_pickle(dim_mls_filepath)
dim_og_df = pd.read_pickle(dim_og_filepath)

save_dir = os.path.join(rans_dir, 'kde')
#config setup and plotting
save = False
bds = (0.0, 100.0)
sets = ['FS8']
feature_sets = {}

feature_sets = {}
for fs in sets:
    config = {
        "features": {
            "input": fs,      # or "FS3b", etc.
            "trim_z": True       # Optional z-trimming
        }
    }
    feature_sets[fs] = get_feature_set(config)
    multi_feature_kde_plot(og_df, config=config, grad_type='OG',
                       d_nd=d_nd, save_dir=save_dir,
                       case_name='Case1', bounds=bds, save=save)
    multi_feature_kde_plot(mls_df, config=config, grad_type='MLS',
                       d_nd=d_nd, save_dir=save_dir,
                       case_name='Case1', bounds=bds, save=save)


mls_df = nd_mls_df
og_df = nd_og_df
length = len(mls_df['Uy'])
print(f'MLS df len: {length}')
length = len(og_df['Uy'])
print(f'og df length: {length}')

ux_rng = mls_df['Ux'].quantile(0.99) - mls_df['Ux'].quantile(0.01)    
ux =mls_df['Ux']
print(f'\nRange of Ux: {ux_rng} \n')
print(f'\n Describe Ux: {ux.describe()}\n')
'''

from scipy.signal import find_peaks
from scipy.stats import gaussian_kde


def is_bimodal(series, min_prominence=0.01):
    data = series.dropna() if hasattr(series, 'dropna') else series[~np.isnan(series)]

    # Kernel Density Estimation
    kde = gaussian_kde(data)
    xs = np.linspace(data.min(), data.max(), 10000)
    ys = kde(xs)

    # Find peaks in the KDE curve
    peaks, _ = find_peaks(ys, prominence=min_prominence)

    # Return True if 2 or more significant peaks are found
    return len(peaks) >= 2


def get_tail_clip_values(df, features, kurt_threshold=1.5, low_clip_small=0.02, high_clip_small=0.05, low_clip_large=0.02, high_clip_large=0.1):
    
    clip_values = {}

    for feat in features:
        series = df[feat]
        kurt = series.kurtosis()
        skew = series.skew()
        
        if is_bimodal(df[feat]):
            pct = 0.005
            print(f"{feat} is bimodal, using slim, {pct} trim bounds")
            if feat == 'Ux':
                low_clip_pct = 0.0
                high_clip_pct = 0.0
            elif feat == 'Adv_xx' or feat == 'Adv_x' or feat == 'R_xy' or feat == 'S_xy':
                low_clip_pct = 0.0
                high_clip_pct = 0.05
            elif feat == 'comp_yx' or feat == 'comp_y' or feat == 'R_yx' or feat =='ddUy_dx_dy' or feat == 'drho_UxUx_dy':
                low_clip_pct = 0.0
                high_clip_pct = 0.1
            elif feat == 'strain_adv_x':
                low_clip_pct = 0.0
                high_clip_pct = 0.05
            else:
                low_clip_pct = pct
                high_clip_pct = pct
        else:
            if feat == 'Ux':
                print('Ux is passed the else statement')
            # Set base clipping percentages based on kurtosis threshold
            if kurt < kurt_threshold:
                low_clip_pct = low_clip_small
                high_clip_pct = high_clip_small
            else:
                low_clip_pct = low_clip_large
                high_clip_pct = high_clip_large

        # Reverse clip percentages if skew is negative (long left tail)
        if feat != 'Ux' and skew < 0:
            low_clip_pct, high_clip_pct = high_clip_pct, low_clip_pct
        
      

        # Calculate the actual values at the quantiles
        low_val = series.quantile(low_clip_pct)
        high_val = series.quantile(1 - high_clip_pct)
        
        clip_values[feat] = (low_val, high_val)

    return clip_values

'''
#test 1
config1 = {
    "features": {
        "input": "FS2",
        "trim_z": True
    }
}
print('Test1')
fs1 = get_feature_set(config1)
print("Features FS1 (z-trimmed):")
print(sorted(fs1))

features = ['Ux','dUx_dy', 'drho_UxUy_dx', 'R_yx', 'comp_yx', 'R_xy', 'mu_t', 'S_xy', 'dUy_dx', 'drho_UxUx_dy', 'comp_y', 'ddUy_dx_dy', 'Adv_xx', 'Adv_x', 'Uy']

clip_vals = get_tail_clip_values(mls_df, fs1)
print("Clip values just calculated:")
#for feat, (low_val, high_val) in clip_vals.items():
#    print(f"{feat}: Clip lower at {low_val:.4f}, clip upper at {high_val:.4f}")
    '''
    
import matplotlib.pyplot as plt
import seaborn as sns

def plot_features_with_clip_bounds(df, feature_list, clip_bounds_dict):
    n_features = len(feature_list)
    ncols = 3
    nrows = (n_features + ncols - 1) // ncols  # Ceiling division for subplot rows

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten()

    for i, feature in enumerate(feature_list):
        ax = axes[i]
        sns.kdeplot(df[feature].dropna(), ax=ax, fill=True, color='skyblue')
        ax.set_xlim(df[feature].min(), df[feature].max())
        if feature == 'Ux':
            rng = df['Ux'].max() - df['Ux'].min()
            print(f'Ux Range: {rng}')
        if feature in clip_bounds_dict:
            
            lower_clip, upper_clip = clip_bounds_dict[feature]
    
            # Shade region between lower_clip and upper_clip
            ax.axvspan(lower_clip, upper_clip, color='orange', alpha=0.3, label='Clip Bounds')
            
            # Optionally, draw vertical lines at clip bounds for clarity
            ax.axvline(lower_clip, color='red', linestyle='--')
            ax.axvline(upper_clip, color='red', linestyle='--')
        ax.set_title(feature)
        ax.legend()

    # Remove any empty subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()
'''
plot_features_with_clip_bounds(mls_df, fs1, clip_vals)
'''

def get_global_clip_bounds(clip_bounds_dict, feature_list=None):
    if feature_list is None:
        feature_list = list(clip_bounds_dict.keys())

    all_mins = []
    all_maxs = []

    for feat in feature_list:
        if feat in clip_bounds_dict:
            low, high = clip_bounds_dict[feat]
            all_mins.append(low)
            all_maxs.append(high)
    
    global_min = min(all_mins)
    global_max = max(all_maxs)

    return global_min, global_max

#gmin, gmax = get_global_clip_bounds(clip_vals)

