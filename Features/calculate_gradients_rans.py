# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 11:15:04 2025

@author: eoporter
"""
# so the methodology of gradients is as follow:
''' at all costs, we do not want to do any interpolation on this data, 
    it skews the original and you can lose critical information about the system within
    functions like np.gradient or scipy gradient can perform grid based differntiation.
    However, you must construct a uniform grid from the unique x and y values of your mesh.
    It is tricky when one region of your mesh is more refined than another and many new points 
    will be added into your domain. This is when you have to integrate to fill the holes of your data. 
    This is where you are at the mercy of numerical methods and differentiation efforts.'''
    
''' As a workaround, I will be using the Moving Least Squares (MLS) gradient 
    estimation method, which is highly robust to irregularly spaced data. 
    The core idea of MLS is to locally approximate the scalar field around a 
    point of interest by fitting a linear function using its neighboring points 
    within a specified radius. 

    For each point, we:
      - Identify neighboring points within a radius using a spatial tree (e.g., KDTree).
      - Fit a plane (in 2D) or hyperplane (in higher dimensions) of the form:
            f(x, y) ≈ a + b*(x - x₀) + c*(y - y₀)
        where (x₀, y₀) is the point of interest.
      - Solve a weighted least squares problem to determine the coefficients.
        The weights decay with distance — typically using a Gaussian kernel.
      - The coefficients (b, c) correspond to the gradient ∇f at that point.

    This approach avoids interpolation by working directly on the original mesh, 
    respects local data density. It is only one step as well, so instead of interpolating
    and then taking the gradient, two steps where lots of information can be lost, we perform
    this method to reduce the amount of deviation from the true data and reduce errors that we could introduce
    This is useful for the model because misrepresenting the data can make it harder for models to converge'''
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 11:48:20 2025

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
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import os


#Dynamically resolve project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add to system path if not already
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(PROJECT_ROOT)
from Shear_mixing.boundary_conditions import BCs


idx = '2'
num='1'

case=f'Case{idx}'
gtype = 'MLS'
grad_type = f'{gtype}_grads'
first_second = 'second'
data_path = dir_path = os.path.join(PROJECT_ROOT,'Data', 'Shear_mixing', 'RANS')
#change the dir path based on what kind of grads you are calculating
dir_path = os.path.join(data_path, grad_type, first_second)
print(os.listdir(dir_path))
data_name = case + f'_{gtype}_grad2.pkl'

load_path = os.path.join(dir_path, data_name)
df = pd.read_pickle(load_path)
print(df.columns)
bds_cx = (df['Cx'].min(), df['Cx'].max())
ln_cx = len(df['Cx'])
print(f'length of cx and bounds:{bds_cx}, {ln_cx} ')


def mls_gradient2d(coords, values, radius, kernel='gaussian', eps=1e-8):
    """
    Estimate gradients using Moving Least Squares (MLS) in 2D.
    
    Parameters:
    - coords: (N, 2) array of (x, y) coordinates.
    - values: (N,) array of scalar field values.
    - radius: float, radius for local neighborhood.
    - kernel: str, type of weight kernel ('gaussian' or 'inverse').
    - eps: small constant to prevent division by zero.
    
    Returns:
    - grads: (N, 2) array of gradient vectors (df/dx, df/dy)
    """
    tree = KDTree(coords)
    grads = np.zeros_like(coords)
    N = len(coords)

    for i, (xi, yi) in enumerate(coords):
        idx = tree.query_ball_point([xi, yi], radius)
        if i % 1000 == 0 or i == N - 1:
            print(f"[MLS] Processing point {i+1} / {N}")
        if len(idx) < 3:
            grads[i] = np.nan  # not enough neighbors to fit plane
            continue

        neighbors = coords[idx]  # (M, 2)
        displacements = neighbors - [xi, yi]  # (M, 2)
        f_neighbors = values[idx]  # (M,)

        if kernel == 'gaussian':
            dists2 = np.sum(displacements**2, axis=1)
            weights = np.exp(-dists2 / (radius**2 + eps))
        elif kernel == 'inverse':
            dists = np.linalg.norm(displacements, axis=1)
            weights = 1 / (dists + eps)
        else:
            weights = np.ones(len(idx))

        # Design matrix: [dx_j, dy_j]
        A = displacements
        W = np.diag(weights)
        b = f_neighbors - f_neighbors.mean()

        # Solve weighted least squares: (A^T W A) x = A^T W b
        try:
            ATA = A.T @ W @ A
            ATb = A.T @ W @ b
            grad = np.linalg.solve(ATA, ATb)
            grads[i] = grad
        except np.linalg.LinAlgError:
            grads[i] = np.nan  # singular matrix, skip

    return grads

def make_grads(df, coords, grad_feats, radius, compute_dz=False):
    """
    Compute gradients for features in grad_feats.

    Parameters:
    - df: DataFrame containing features.
    - coords: Nx2 array of (x, y) coordinates.
    - grad_feats: list of feature names (strings) to compute gradients for.
    - radius: neighborhood radius for MLS.
    - compute_dz: bool, whether to compute dz gradients (if False, fill zeros).

    Returns:
    - df updated with gradients: d{feat}_dx, d{feat}_dy, d{feat}_dz.
    """
    computed = {}  # to track which features are computed, for second derivs etc.

    for feat in grad_feats:
        # Check if feature exists in df
        if feat not in df.columns:
            print(f"[WARN] Feature '{feat}' not found in df, skipping.")
            continue
        
        values = df[feat].values
        print(f'On feat: {feat}')
        # Compute MLS gradients in x,y
        grad_xy = mls_gradient2d(coords, values, radius)
        df[f'd{feat}_dx'] = grad_xy[:, 0]
        df[f'd{feat}_dy'] = grad_xy[:, 1]
        
        # For dz, either zeros or compute if requested
        if compute_dz:
            # Implement dz gradient calculation here if possible
            # For now, just zeros
            df[f'd{feat}_dz'] = np.zeros_like(values)
        else:
            df[f'd{feat}_dz'] = np.zeros_like(values)
        
        # Add to computed dict for potential second derivative computation
        computed[feat] = (f'd{feat}_dx', f'd{feat}_dy', f'd{feat}_dz')
    
    return df, computed


'''
#initialize args for function
alpha = 1.0
ap = str(alpha)
l_ref = BCs[case]['Reference']['x_ref']
rads = alpha*l_ref
'''
first_grad_feats = ['Ux', 'Uy', 'Uz', 'rho', 'p', 'T']
second_grad_feats = ['dUx_dx', 'dUx_dy', 'dUy_dx', 'dUy_dy', 'dUz_dx', 'dUz_dy',
              'Tao_xx', 'Tao_xy', 'Tao_yy', 'Tao_zz', 'Tao_xz', 'Tao_yz',
              'rho_UxUx', 'rho_UxUy', 'rho_UxUz', 'rho_UyUy', 'rho_UyUz', 'rho_UzUz']
feats = {'first': first_grad_feats, 'second': second_grad_feats}


def trim_by_min_x(df, min_x):
    return df[df['Cx'] >= min_x].copy()

#df_a1, computed_a1 = make_grads(df, coords, second_grad_feats,radius = alpha*l_ref, compute_dz=True)  

# iterate to find best one
'''
alphas = [0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0, 1.2]
metrics = {}
for alph in alphas:
    print(f'On alpha: {alph}')
    metrics[alph] = {}
    a_dic = metrics[alph]
    df_a, comp_a = make_grads(df, coords, second_grad_feats, radius=alph*l_ref, compute_dz=False)
    for comp in comp_a:
        a_dic[comp] = {}
        stat_dic = a_dic[comp]
        stat_dic['mean'] =df_a[comp].mean()
        stat_dic['std'] = df_a[comp].std()
        stat_dic['metric'] = df_a[comp].std() / np.abs(df_a[comp].mean())


records = []
for alpha, comp_dict in metrics.items():
    for comp, stats in comp_dict.items():
        records.append({
            'alpha': alpha,
            'component': comp,
            'mean': stats['mean'],
            'std': stats['std'],
            'metric': stats['metric']
        })

# Convert to DataFrame
metrics_df = pd.DataFrame(records)

# Example: view metrics sorted by component
print(metrics_df.sort_values(['component', 'alpha']))
for comp in metrics_df['component'].unique():
    subset = metrics_df[metrics_df['component'] == comp]
    plt.plot(subset['alpha'], subset['metric'], marker='o', label=comp)

plt.xlabel('Alpha (radius / L_ref)')
plt.ylabel('Smoothness metric (std / mean)')
plt.title('Gradient Smoothness vs. Alpha')
plt.legend()
plt.grid(True)
plt.show()
'''

#save paths

#trim so not calculating grads in odd area
l_ref = BCs[case]['Reference']['x_ref']
df = trim_by_min_x(df, min_x=-2*l_ref)
coords = np.vstack((df['Cx'], df['Cy'])).T
#now call grad function
first_second = 'second'
gtype = 'MLS'
save_path = os.path.join(data_path, f'{gtype}_grads', first_second)
#first_second = 'first'
if first_second == 'first':
    num = '1'
else:
    num = '2'

alpha = 0.4
x_ref = alpha * l_ref
df, computed = make_grads(df, coords, feats[first_second], x_ref, compute_dz=False)
save_filename= f'{case}_{gtype}_grad3.pkl'
save_results=True
if save_results:
    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, save_filename)
    df.to_pickle(out_path)
    print(f"[SAVED] Gradient-augmented DataFrame saved to:\n{out_path}")

    
def contourf_plot(coords, field, cmap='jet', title='', show=True, save_path=None,
                  levels=100, log_scale=False, log_eps=1e-8):
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
    - log_scale: whether to plot log10 of the field (with sign preserved)
    - log_eps: small constant to avoid log(0)
    """

    x = coords[:, 0]
    y = coords[:, 1]
    z = field.copy()

    if log_scale:
        z = np.sign(z) * np.log10(np.abs(z) + log_eps)
        title = f"log10({title})" if title else "log10(Contourf Plot)"

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
    ax.set_title(title)
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

def plot_all_features(df, coords, save_dir=None, show=False, cmap='jet', levels=100, log_scale=False, grad_type=''):
    """
    Loop through all numeric features in the DataFrame and plot them.

    Parameters:
    - df: pandas DataFrame containing the data
    - coords: Nx2 array of (x, y) coordinates
    - save_dir: optional directory to save the plots
    - show: whether to display plots
    - cmap: colormap to use
    - levels: contour levels
    """
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    grad_feats = []
    if grad_type == 'second':
        grad_feats = second_grad_feats
    else: 
        grad_feats = first_grad_feats
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for col in grad_feats:
        print(f"[PLOT] Plotting {col}")
        field = df[col].values
        save_path = os.path.join(save_dir, f"{col}.png") if save_dir else None
        contourf_plot(coords, field, cmap=cmap, title=col, show=show,
                      save_path=save_path, levels=levels, log_scale=True)



'''
df = trim_by_min_x(df, min_x=0)
print(df.columns)
coords = np.vstack((df['Cx'], df['Cy'])).T       
plot_all_features(df, coords, save_dir=False, show=True)
'''
