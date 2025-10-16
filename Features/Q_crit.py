from PreProcess.Load_and_norm import load_dfs
import pandas as pd
import pickle as pkl
import os
import sys
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
print(project_root)


idx = '2'
case = f'Case{idx}'
gtype = 'MLS'
dnd = 'dim'
fov = '1'
filename = f'{case}_{dnd}_FOV{fov}.pkl'
# Define key paths
big_data_dir = os.path.join(project_root, 'Data')
rans_dir = os.path.join(big_data_dir, 'Shear_mixing', 'RANS')
training_dir = os.path.join(rans_dir, 'training')
ddir = os.path.join(training_dir, 'MLS')
load_path = os.path.join(ddir, filename)

#filepath = os.path.join(source_dir, grads_type ,first_second ,filename)
df = pd.read_pickle(load_path)

print(df.columns)


def get_rot_adv_mag(df):
    rax, ray, raz = df['rot_adv_x'], df['rot_adv_y'], df['rot_adv_z']
    ra_mag = np.sqrt(rax ** 2 + ray ** 2 + raz ** 2)
    df['Rot_adv_mag'] = ra_mag
    return df
def get_strain_adv_mag(df):
    sax, say, saz = df['strain_adv_x'], df['strain_adv_y'], df['strain_adv_z']
    sa_mag = np.sqrt(sax ** 2 + say ** 2 + saz ** 2)
    df['Strain_adv_mag'] = sa_mag
    return df

def get_strain_mag(df):
    sxx, syy, szz = df['S_xx'], df['S_yy'], df['S_zz']
    sxy, sxz, syz = df['S_xy'], df['S_xz'], df['S_yz']
    s_mag = np.sqrt(sxx**2 + syy**2 + szz**2 + 2*(sxy**2 + sxz**2 + syz**2))
    df['Strain_mag'] = s_mag
    return df

def get_rot_mag(df):
    rxy, rxz, ryz = df['R_xy'], df['R_xz'], df['R_yz']
    r_mag = np.sqrt(2*(rxy**2 + rxz**2 + ryz**2))
    df['Rot_mag'] = r_mag
    return df

def get_q_norm(df, eps=1e-12):
    """Normalized Q-criterion: (‖R‖² - ‖S‖²)/(‖R‖² + ‖S‖²)."""
    s2 = df['Strain_mag']**2
    r2 = df['Rot_mag']**2
    df['Q_norm'] = (r2 - s2) / (r2 + s2 + eps)
    return df

def get_strain_rot_balance(df, eps=1e-12):
    """Symmetric balance and advection preferences."""
    s, r = df['Strain_mag'], df['Rot_mag']
    sa, ra = df['Strain_adv_mag'], df['Rot_adv_mag']
    df['S_vs_R'] = (s - r) / (s + r + eps)
    df['Str_pref_adv'] = sa / (sa + ra + eps)
    df['Rot_pref_adv'] = ra / (sa + ra + eps)
    return df

def get_cos_theta_adv(df, eps=1e-12):
    """Directional alignment between strain_adv and rot_adv vectors."""
    num = (df['rot_adv_x']*df['strain_adv_x'] +
           df['rot_adv_y']*df['strain_adv_y'] +
           df['rot_adv_z']*df['strain_adv_z'])
    den = df['Rot_adv_mag'] * df['Strain_adv_mag'] + eps
    df['cos_theta_adv'] = num / den
    return df

def get_lambda2_and_ci(df, eps=1e-12):
    """
    2D λ2 and swirling strength λ_ci from velocity gradients.
    Requires dUx_dx, dUx_dy, dUy_dx, dUy_dy to exist in df.
    """
    a = df['dUx_dx']; b = df['dUx_dy']
    c = df['dUy_dx']; d = df['dUy_dy']

    # Strain & rotation for 2x2 system
    Sxx = a; Syy = d; Sxy = 0.5*(b+c)
    Omxy = 0.5*(b-c)

    # J = S^2 + Ω^2
    Jxx = Sxx**2 + Sxy**2 + Omxy**2
    Jyy = Syy**2 + Sxy**2 + Omxy**2
    Jxy = Sxy*(Sxx + Syy)

    tr = Jxx + Jyy
    det = Jxx*Jyy - Jxy**2
    disc = np.maximum(tr**2/4 - det, 0.0)
    lam1 = tr/2 + np.sqrt(disc)
    lam2 = tr/2 - np.sqrt(disc)
    df['lambda2_crit'] = lam2  # λ2 < 0 ⇒ vortex core

    # Swirling strength (imag part of eigenvalues of velocity gradient A)
    trA = a + d
    discA = (a - d)**2 + 4*b*c
    df['lambda_ci'] = 0.5*np.sqrt(np.maximum(-discA, 0.0))
    return df


def get_excess_strain_rot(df):
    r_mag, s_mag = df['Rot_mag'], df['Strain_mag']
    ra_mag, sa_mag = df['Rot_adv_mag'], df['Strain_adv_mag']
    ex_str_rot = (s_mag - r_mag )/(s_mag+r_mag)
    ex_str_rot_adv = (sa_mag - ra_mag) / (sa_mag+ra_mag)
    df['ex_str_rot'] = ex_str_rot
    df['ex_str_rot_adv'] = ex_str_rot_adv
    return df

def quick_feat_calc(df):
    df = get_rot_mag(df)
    df = get_strain_mag(df)
    df = get_rot_adv_mag(df)
    df = get_strain_adv_mag(df)
    df = get_excess_strain_rot(df)

    # New extras
    df = get_q_norm(df)
    df = get_strain_rot_balance(df)
    df = get_cos_theta_adv(df)
    df = get_lambda2_and_ci(df)

    return df





from Plotting.plot_single_feature import tricontour_plot, tripcolor_plot
import matplotlib.pyplot as plt

features = [ 'strain_adv_x', 'strain_adv_y', 'strain_adv_z',
            'rot_adv_x', 'rot_adv_y', 'rot_adv_z', 'Strain_mag',
             'Rot_mag', 'Strain_adv_mag', 'Rot_adv_mag',
             'ex_str_rot', 'ex_str_rot_adv']
df_calc = quick_feat_calc(df)
feats = ['ex_str_rot', 'ex_str_rot_adv']
new_features = [
    'Q_norm',
    'S_vs_R',
    'Str_pref_adv',
    'Rot_pref_adv',
    'cos_theta_adv',
    'lambda2_crit',
    'lambda_ci']

for feat in new_features:
    tripcolor_plot(df_calc, feat, cmap='jet')
