# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 16:14:22 2025

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

print(PROJECT_ROOT)
from Data.Shear_mixing.boundary_conditions import BCs

idx = '1'
case = f'Case{idx}'
rans_filename = f'{case}_pre_grads_1.pkl'
grads_type = 'og_grads'


def load_case_data_by_prefix(directory, prefix='Case1'):
    case_data = {}

    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith('.pkl'):
            fov_label = filename.split('_')[2]
            label = fov_label.split('.')[0]   # e.g., 'FOV1'
            file_path = os.path.join(directory, filename)
            try:
                df = pd.read_pickle(file_path)
                case_data[label] = df
                print(f"✅ Loaded {filename} as '{label}'")
            except Exception as e:
                print(f"❌ Failed to load {filename}: {e}")

    return case_data
def combine_fovs(fov_dict):
    dfs = []
    for fov_label, df in fov_dict.items():
        df_copy = df.copy()
        df_copy['FOV'] = fov_label
        dfs.append(df_copy)

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df
def get_rans_slice(df, y_bounds, x_bounds):   
    y_min, y_max = y_bounds
    x_min, x_max = x_bounds

    return df[
                (df['Cy'] >= y_min) & (df['Cy'] <= y_max) &
                (df['Cx'] >= x_min) & (df['Cx'] <= x_max)]
# Set project root and directory structure
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
print(project_root)
# Define key paths
big_data_dir = os.path.join(project_root, 'data')
source_dir = os.path.join(big_data_dir, 'Shear_mixing')
exp_dir = os.path.join(source_dir, 'processed_exp')
rans_dir = os.path.join(source_dir, 'RANS', grads_type)
rans_path = os.path.join(rans_dir, rans_filename)
#load rans and exp for same case
# Load data
rans = pd.read_pickle(rans_path)
print(rans.columns)
pref = 'processed_'+ case
exp_dict = load_case_data_by_prefix(exp_dir, prefix=pref)
#print(exp_dict.keys())
print(exp_dict['FOV1'].columns)
# Convert experimental coordinates to meters

# get fov bounds
fov_bounds = {}
for i, df in exp_dict.items():
    cx_min, cx_max = min(df['Cx']), max(df['Cx'])
    cy_min, cy_max = min(df['Cy']), max(df['Cy'])
    fov_bounds[i] = {'x_bounds': (cx_min, cx_max),
                     'y_bounds': (cy_min, cy_max),
                     'width': cx_max - cx_min,
                     'height': cy_max - cy_min, 
        }
#print(f'Fov Bounds: \n {fov_bounds}')
# --- RANS Domain Bounds ---
rans_bounds_x = (rans['Cx'].min(), rans['Cx'].max())
rans_bounds_y = (rans['Cy'].min(), rans['Cy'].max())
#print(f'RANS x bounds: {rans_bounds_x}')
#print(f'RANS y bounds: {rans_bounds_y}')


# --- Optional Trim by x start (if needed) ---
x_start_threshold = 0.0
rans['Cx'] = rans['Cx'] - x_start_threshold
rans_coords = np.vstack((rans['Cx'], rans['Cy'])).T
rans['U_mag'] = np.sqrt(np.square(rans['Ux']) + np.square(rans['Uy']) + np.square(rans['Uz']))

#add tke real quick
rans['tke'] = 0.5*(rans['uu'] + rans['vv'] + rans['ww'])
rans['ww_p'] = 2 * rans['k'] - rans['uu'] - rans['vv']
rans['ww_pp'] = rans['ww'] - rans['ww_p']
rans['tke_p'] = rans['k'] - rans['tke']
# --- Extract FOV Slice ---
fov_key = 'FOV1'
bounds = fov_bounds[fov_key]
xbound_st = fov_bounds['FOV1']['x_bounds'][0]
xbound_end = fov_bounds['FOV1']['x_bounds'][1]
#xbound_end = fov_bounds['FOV3']['x_bounds'][1]

ybounds = fov_bounds[fov_key]['y_bounds']
xbounds = (xbound_st, xbound_end) 
rans_slice = get_rans_slice(rans, ybounds, xbounds)
xmin = min(rans_slice['Cx'])
print(f'Min rans slice x: {xmin}')
print(f'Min bounds x: {xbound_st}')
slice_coords = np.vstack((rans_slice['Cx'], rans_slice['Cy'])).T

# --- Plot ---
def tripcolor_plot(coords, field, cmap='viridis', title='', show=True, save_path=None, mask_thresh=1.0):
    
    x = coords[:, 0]
    y = coords[:, 1]

    triang = tri.Triangulation(x, y)

    # Mask triangles with long edges (bad triangles or sparse regions)
    triangles = triang.triangles
    pts = coords[triangles]  # shape (n_tri, 3, 2)

    # Compute edge lengths
    a = np.linalg.norm(pts[:, 0] - pts[:, 1], axis=1)
    b = np.linalg.norm(pts[:, 1] - pts[:, 2], axis=1)
    c = np.linalg.norm(pts[:, 2] - pts[:, 0], axis=1)
    max_edge = np.max(np.stack([a, b, c], axis=1), axis=1)

    # Mask triangles with max edge length > threshold
    mask = max_edge > mask_thresh
    triang.set_mask(mask)

    fig, ax = plt.subplots(figsize=(8, 6))
    tpc = ax.tripcolor(triang, field, shading='flat', cmap=cmap)
    
    
    fig.colorbar(tpc, ax=ax)
    ax.set_title(title or "Tripcolor Plot")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    #ax.set_aspect('equal',adjustable='box' )
    ax.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

    return fig
def sample_and_plot_tripcolor(df, field,shifted=False, n=10, **kwargs):
    df_sampled = df.iloc[::n].copy()
    if shifted:
        cx = df_sampled['Cx_shifted']
    else:
        cx = df_sampled['Cx']
    coords = np.vstack((cx, df_sampled['Cy'])).T
    values = df_sampled[field]
    
    return tripcolor_plot(coords, values, **kwargs)
from scipy.interpolate import griddata

def contourf_plot(coords, field, cmap='jet', title='', show=True, save_path=None, levels=100):
    """
    Create a filled contour plot from scattered data.

    Parameters:
    - coords: Nx2 array of (x, y) coordinates
    - field: field values at each (x, y)
    - cmap: colormap
    - title: plot title
    - show: whether to display the plot
    - save_path: optional path to save image
    - levels: number of contour levels
    """

    x = coords[:, 0]
    y = coords[:, 1]
    z = field

    # Create a regular grid to interpolate onto
    xi = np.linspace(x.min(), x.max(), 300)
    yi = np.linspace(y.min(), y.max(), 300)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate using griddata
    zi = griddata((x, y), z, (xi, yi), method='linear')

    # Plot
    fig, ax = plt.subplots(figsize=(6, 8))
    cf = ax.contourf(xi, yi, zi, levels=levels, cmap=cmap)

    fig.colorbar(cf, ax=ax)
    ax.set_title(title or "Contourf Plot")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

    return fig

    

# --- Visualize RANS Field in FOV ---
levels = 100
feat = 'rho_UyUy'
contourf_plot(rans_coords, rans[feat], cmap='jet', title = f'Full RANS {feat}', show=True, levels=levels)
#feat = 'Tao_yy'
contourf_plot(slice_coords, rans_slice[feat], cmap='jet', title=f'RANS FOV {fov_key} {feat}', show=True, levels=levels)
#make plot for exp now
feat = 'dUx_dz'
contourf_plot(slice_coords, rans_slice[feat], cmap='jet', title=f'RANS FOV {fov_key} {feat}', show=True, levels=levels)
feat = 'dUy_dz'
contourf_plot(slice_coords, rans_slice[feat], cmap='jet', title=f'RANS FOV {fov_key} {feat}', show=True, levels=levels)
feat = 'dUz_dz'
contourf_plot(slice_coords, rans_slice[feat], cmap='jet', title=f'RANS FOV {fov_key} {feat}', show=True, levels=levels)
feat = 'dp_dz'
contourf_plot(slice_coords, rans_slice[feat], cmap='jet', title=f'RANS FOV {fov_key} {feat}', show=True, levels=levels)
feat = 'dUx_dx'
contourf_plot(slice_coords, rans_slice[feat], cmap='jet', title=f'RANS FOV {fov_key} {feat}', show=True, levels=levels)
feat = 'dUx_dy'
contourf_plot(slice_coords, rans_slice[feat], cmap='jet', title=f'RANS FOV {fov_key} {feat}', show=True, levels=levels)

dfs = []
for fov_label, df in exp_dict.items():
    df_copy = df.copy()
    leng = len(df['Cx'])
    print(f'FOV {fov_label} len: {leng}')
    df_copy['FOV'] = fov_label  # mark the source FOV
    dfs.append(df_copy)

# Concatenate all into one dataframe
from Data.Shear_mixing.boundary_conditions import BCs
bc = BCs[case]
delta_u = bc['Reference']['delta_U']
exp = pd.concat(dfs, ignore_index=True)
print(f'Length experimental data:')
fov = get_rans_slice(exp, ybounds, xbounds) 
fov_coords = np.vstack((fov['Cx'], fov['Cy'])).T
feat = 'TR_uvv'
ture = fov[feat]
contourf_plot(fov_coords, ture, cmap='jet', title=f'Exp {feat}', show=True, levels=levels)

# Define zoom window (±2 mm = ±0.002 m)
zoom_range = 0.005

# Assuming your shifted RANS data has 'Cx' aligned to origin
origin_x = 0.0  # the zero reference after shifting
origin_y = 0.0  # usually zero in Cy unless you want to offset here too

# Define bounds for zoom box
x_min, x_max = - zoom_range,  zoom_range
y_min, y_max =  - zoom_range,  zoom_range

# Slice RANS data in this zoom region
rans_zoom = rans[
    (rans['Cx'] >= x_min) & (rans['Cx'] <= x_max) &
    (rans['Cy'] >= y_min) & (rans['Cy'] <= y_max)
    ]

'''
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Coordinates and values from RANS slice
x_rans = rans_slice['Cx'].values
y_rans = rans_slice['Cy'].values

feat_rans = 'dUx_dz'
Ux_rans = rans_slice[feat_rans].values

# Coordinates and values from experimental slice

x_exp = fov['Cx'].values
y_exp = fov['Cy'].values
feat_exp = 'Skew_u'
U_exp = fov[feat_exp].values
# Interpolate experimental data onto RANS points
points_exp = np.column_stack((x_exp, y_exp))
values_exp = U_exp
points_rans = np.column_stack((x_rans, y_rans))
U_exp_on_rans = griddata(points_exp, values_exp, points_rans, method='linear')

# Create grid over the RANS domain
size = 1000
xi = np.linspace(x_rans.min(), x_rans.max(), size)
yi = np.linspace(y_rans.min(), y_rans.max(), size)
xi, yi = np.meshgrid(xi, yi)

# Interpolate RANS Ux and experimental data onto the grid
zi_rans = griddata(points_rans, Ux_rans, (xi, yi), method='linear')
zi_exp = griddata(points_rans, U_exp_on_rans, (xi, yi), method='linear')

# Plot RANS TKE
fig1, ax1 = plt.subplots(figsize=(7, 5))
cf1 = ax1.contourf(xi, yi, zi_rans, levels=100, cmap='jet')
ax1.set_title(f'RANS {feat_rans}')
cbar1 = fig1.colorbar(cf1, ax=ax1, orientation='vertical')
cbar1.set_label(f'{feat_rans}')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)

# Plot Experimental TKE (interpolated)
fig2, ax2 = plt.subplots(figsize=(7, 5))
cf2 = ax2.contourf(xi, yi, zi_exp, levels=100, cmap='jet')
ax2.set_title(f'Exp {feat_exp} interpolated on RANS grid')
cbar2 = fig2.colorbar(cf2, ax=ax2, orientation='vertical')
cbar2.set_label(f'{feat_exp}')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.grid(True)

plt.show()'''