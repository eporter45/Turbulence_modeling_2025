# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 11:52:10 2025

@author: eoporter
"""
import sys
import os
import pandas as pd
# Set project root and directory structure
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
print(project_root)
# Define key paths


idx = '2'
case = f'Case{idx}'
dnd = 'dim'
load_dir = os.path.join(project_root, 'Shear_mixing', 'EXP', 'train_exp', dnd)

fovs = {}

for i in range(4):
    j = i+1
    filename = f'{dnd}_{case}_FOV{j}.pkl'
    fovs[f'FOV{j}'] = pd.read_pickle(os.path.join(load_dir, filename))

def concatenate_dfs(dfs_dict):
    if not dfs_dict:
        print("No DataFrames to concatenate.")
        return pd.DataFrame()  # return empty DataFrame
    
    combined_df = pd.concat(dfs_dict.values(), ignore_index=True)
    print(f"Concatenated {len(dfs_dict)} DataFrames into one with {combined_df.shape[0]} rows.")
    return combined_df

combined = concatenate_dfs(fovs)

import matplotlib.pyplot as plt
import matplotlib.tri as tri



def tripcolor_plot(df, field, cmap='viridis', show=True, figsize=(10,6), title=None):
    """
    """
    x = df['Cx'].values
    y = df['Cy'].values
    z = df[field].values

    # Create a triangulation from the scattered points
    triang = tri.Triangulation(x, y)

    fig, ax = plt.subplots(figsize=figsize)
    tpc = ax.tripcolor(triang, z, shading='gouraud', cmap=cmap)
    fig.colorbar(tpc, ax=ax, label=field)
    ax.set_xlabel('Cx')
    ax.set_ylabel('Cy')
    if title:
        ax.set_title(title)
    plt.axis('equal')
    plt.grid(False)
    if show:
        plt.show()

    return fig


feats = ['uu', 'vv', 'ww', 'uv','a_xx', 'a_xy','a_yy', 'a_zz', 'tke']
feats = ['U', 'V', 'W', 'tke']
for feat in feats:
    fig = tripcolor_plot(combined, feat, cmap='viridis', title=f'Case{idx} Full - {feat}')