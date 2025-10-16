# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 11:34:00 2025

@author: eoporter
"""
    
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd

### Helper Funcitons

# put near the top with other helpers if you like
RST6 = ['uu','uv','vv','uw','vw','ww']
AIJ6 = ['a_xx','a_xy','a_yy','a_xz','a_yz','a_zz']

def _detect_family(output_features):
    low = [c.lower() for c in output_features]
    if any(n.startswith('a_') for n in low): return 'aij'
    if any(n.startswith('b_') for n in low): return 'bij'
    return 'rst'

def _six_names_for_family(output_features):
    fam = _detect_family(output_features)
    if fam == 'aij': return [nm for nm in AIJ6 if nm in output_features]
    return [nm for nm in RST6 if nm in output_features]

def _diag_names_for_family(output_features):
    fam = _detect_family(output_features)
    return (['a_xx','a_yy','a_zz'] if fam == 'aij' else ['uu','vv','ww'])

def get_k_max(config, y_train):
    """
    Max |k| or |tke| across TRAIN. Returns None if k/tke not present.
    """
    feats = config['features']['output']
    low = [f.lower() for f in feats]
    if 'tke' not in low and 'k' not in low:
        return None
    k_name = feats[low.index('tke')] if 'tke' in low else feats[low.index('k')]
    import numpy as np
    k_vals = np.concatenate([df[k_name].values for df in y_train.values()], axis=0)
    k_max = float(np.max(np.abs(k_vals)))
    return k_max if k_max > 0 else 1.0



def get_frob(config, y_train):
    """
    Max Frobenius norm across TRAIN for the 6-tensor columns ONLY.
    Off-diagonals counted twice, diagonals once. Excludes k/tke.
    """
    output_feats = config['features']['output']
    six_names = _six_names_for_family(output_feats)
    diag_names = set(_diag_names_for_family(output_feats))

    import numpy as np
    max_frobs = []
    for df in y_train.values():
        arr = df[six_names].values  # (N,6)
        frob_sq = np.zeros(arr.shape[0])
        for i, feat in enumerate(six_names):
            contrib = arr[:, i] ** 2
            if feat not in diag_names:
                contrib *= 2.0
            frob_sq += contrib
        max_frobs.append(np.sqrt(frob_sq).max())
    max_frob = float(abs(np.max(max_frobs)))
    if max_frob == 0:
        raise ValueError("[normalize_data] Max Frobenius norm is zero; cannot normalize.")
    return max_frob


def is_bimodal(col, min_prominence=0.01):
    data = col.dropna() if hasattr(col, 'dropna') else col[~np.isnan(col)]

    if len(data) < 2:
        print(f"[INFO] Not enough data to test bimodality for column: {getattr(col, 'name', 'unknown')}")
        return False

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
        if df[feat].var() == 0:
            #print(f'Variance is zero for {feat}, skipping')
            continue
        if is_bimodal(df[feat]):
            pct = 0.002
            #print(f"{feat} is bimodal, using slim, {pct} trim bounds")
            if feat == 'Ux':
                low_clip_pct = 0.0
                high_clip_pct = 0.0
            elif feat == 'Adv_xx' or feat == 'Adv_x' or feat == 'R_xy' or feat == 'S_xy':
                low_clip_pct = 0.0
                high_clip_pct = 0.03
            elif feat == 'comp_yx' or feat == 'comp_y' or feat == 'R_yx' or feat =='ddUy_dx_dy' or feat == 'drho_UxUx_dy':
                low_clip_pct = 0.0
                high_clip_pct = 0.1
            elif feat == 'strain_adv_x':
                low_clip_pct = 0.0
                high_clip_pct = 0.03
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

def get_df_bounds(df, features):
    clip_values = get_tail_clip_values(df, features)
    gmin, gmax = get_global_clip_bounds(clip_values, features)
    return gmin, gmax


def get_x_train_bounds(x_train):
    gmaxes, gmins = [], []
    for name, df in x_train.items():
        gmin, gmax = get_df_bounds(df, df.columns)
        gmaxes.append(gmax)
        gmins.append(gmin)
    out_max, out_min = max(gmaxes), min(gmins)
    return out_max, out_min

def normalize_df(df, bounds):
    gmin, gmax = bounds
    feats = list(df.columns)
    for feat in feats:
        df = df.copy()
        ft = df[feat]
        normed = (ft - gmin)/(gmax - gmin)
        df[feat] = normed
    return df


def global_full_norm(x_train, x_test):
    gmax, gmin = get_x_train_bounds(x_train)
    print(f'gmin: {gmin}, gmax: {gmax}')
    x_train_normed, x_test_normed = {}, {}
    for name, df in x_train.items():
        x_train_normed[name] = normalize_df(df, (gmin, gmax))
    for name, df in x_test.items():
        x_test_normed[name] = normalize_df(df, (gmin, gmax))
    return x_train_normed, x_test_normed


def global_column_wise_norm(x_train, x_test, features, return_bounds=False):
    x_train_stacked = pd.concat([df for df in x_train.values()], axis=0)
    clip_bounds = get_tail_clip_values(x_train_stacked, features)
    x_train_normed, x_test_normed = x_train.copy(), x_test.copy()
    
        
    for name, xt in x_train.items():
        xtn = xt.copy()
        for feat in features:
            if not clip_bounds.get(feat, None):
                continue
            else:
                lmin, lmax = clip_bounds[feat]
                xtn[feat] = (xtn[feat] - lmin)/(lmax - lmin)
        x_train_normed[name] = xtn
    for name, xt in x_test.items():
        xtn = xt.copy()
        for feat in features:
            if not clip_bounds.get(feat,None):
                continue
            else:     
                lmin, lmax = clip_bounds[feat]
                xtn[feat] = (xtn[feat] - lmin)/(lmax - lmin)
        x_test_normed[name] = xtn
    if return_bounds:
        return x_train_normed, x_test_normed, clip_bounds
    return x_train_normed, x_test_normed


def log_base_max_transform(x, max_val, epsilon=1e-8):
    return np.log(np.abs(x) + epsilon) / np.log(max_val + epsilon) * np.sign(x)

def ln_base_max_transform(x, max_val, epsilon=1e-8):
    return np.log1p(np.abs(x) / (max_val + epsilon)) * np.sign(x)

def apply_log_normalization(df, max_dict, log_type="log", epsilon=1e-8):
    df_normed = df.copy()
    for feat in df.columns:
        max_val = max_dict[feat]
        if log_type == "log":
            df_normed[feat] = log_base_max_transform(df[feat], max_val, epsilon)
        elif log_type == "ln":
            df_normed[feat] = ln_base_max_transform(df[feat], max_val, epsilon)
        else:
            raise ValueError(f"[ERROR] Unknown log_type '{log_type}', use 'log' or 'ln'.")
    return df_normed

def global_max_log_norm(x_train, x_test, features, log_type="log"):
    # Stack all training data
    stacked = pd.concat([df[features] for df in x_train.values()], axis=0)
    
    # Get robust clip bounds
    clip_bounds = get_tail_clip_values(stacked, features)

    # Use high bound for max scaling
    max_dict = {feat: clip_bounds[feat][1] for feat in features}

    x_train_normed, x_test_normed = {}, {}
    for name, df in x_train.items():
        x_train_normed[name] = apply_log_normalization(df[features], max_dict, log_type)
    for name, df in x_test.items():
        x_test_normed[name] = apply_log_normalization(df[features], max_dict, log_type)

    return x_train_normed, x_test_normed

def local_max_log_norm(x_train, x_test, features, log_type="log"):
    x_train_normed, x_test_normed = {}, {}

    for name, df in x_train.items():
        clip_bounds = get_tail_clip_values(df[features], features)
        max_dict = {feat: clip_bounds[feat][1] for feat in features}
        x_train_normed[name] = apply_log_normalization(df[features], max_dict, log_type)

    for name, df in x_test.items():
        clip_bounds = get_tail_clip_values(df[features], features)
        max_dict = {feat: clip_bounds[feat][1] for feat in features}
        x_test_normed[name] = apply_log_normalization(df[features], max_dict, log_type)

    return x_train_normed, x_test_normed


def get_stats(df, feats, epsilon= 1e-8):
    stats= {}
    for feat in feats:
        col = df[feat]
        q1 = col.quantile(0.25)
        q3 = col.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            iqr = epsilon
        stats[feat] = np.abs(iqr)
    return stats


def local_iqr_norm(x_train, x_test, features):
    xtr_norm, xts_norm = {}, {}
    iqr_stats = {'train': {}, 'test': {}}
    for name, xt in x_train.items():
        xt_norm = xt.copy()
        iqr_stats['train'][name] = get_stats(xt, features)
        for feat in features:
            iqr = iqr_stats['train'][name][feat]
            xt_norm[feat] = xt_norm[feat] / iqr
        xtr_norm[name] = xt_norm
    for name, xt in x_test.items():
        xt_norm = xt.copy()
        iqr_stats['test'][name] = get_stats(xt, features)
        for feat in features:
            iqr = iqr_stats['test'][name][feat]
            xt_norm[feat] = xt_norm[feat] / iqr
        xts_norm[name] = xt_norm
    return xtr_norm, xts_norm


def iqr_norm(x_train, x_test, features):
    x_train_stacked = pd.concat([df for df in x_train.values()], axis=0)
    stacked_stats = get_stats(x_train_stacked, features)
    x_train_normed, x_test_normed = {}, {}
    for name, xt in x_train.items():
        xtn = xt.copy()
        for feat in features:
            iqr = stacked_stats[feat]
            xtn[feat] = (xtn[feat]) / iqr
        x_train_normed[name] = xtn
    for name, xt in x_test.items():
        xtn = xt.copy()
        for feat in features:
            iqr = stacked_stats[feat]
            xtn[feat] = (xtn[feat]) / iqr
        x_test_normed[name] = xtn
    return x_train_normed, x_test_normed



from collections import defaultdict
import pandas as pd

def extract_case_name(fov_key):
    # Assumes keys are formatted like "Case1_FOV1", "Case2_FOV3", etc.
    return fov_key.split('_')[0]

def case_iqr_norm(x_train, x_test, features):
    # Step 1: Group FOVs by case
    case_fov_map = defaultdict(list)
    for name in x_train:
        case_name = extract_case_name(name)
        case_fov_map[case_name].append(x_train[name])
    
    # Step 2: Create case-level concatenated DataFrames
    case_df_map = {case: pd.concat(fov_list, axis=0) for case, fov_list in case_fov_map.items()}
    
    # Step 3: Compute IQR stats per case
    case_iqr_stats = {case: get_stats(df, features) for case, df in case_df_map.items()}
    
    # Step 4: Normalize train FOVs using case IQR
    x_train_normed = {}
    for name, df in x_train.items():
        case = extract_case_name(name)
        stats = case_iqr_stats[case]
        normed_df = df.copy()
        for feat in features:
            iqr = stats[feat]
            normed_df[feat] = normed_df[feat] / iqr
        x_train_normed[name] = normed_df

    # Step 5: Normalize test FOVs using their corresponding case IQR
    x_test_normed = {}
    for name, df in x_test.items():
        case = extract_case_name(name)
        if case not in case_iqr_stats:
            raise ValueError(f"[ERROR] Case '{case}' in test set not found in training set.")
        stats = case_iqr_stats[case]
        normed_df = df.copy()
        for feat in features:
            iqr = stats[feat]
            normed_df[feat] = normed_df[feat] / iqr
        x_test_normed[name] = normed_df

    return x_train_normed, x_test_normed


def case_iqr_global_norm(x_train, x_test, features):
    xtr_norm, xts_norm = case_iqr_norm(x_train, x_test, features)
    xtr2_norm, xts2_norm = global_full_norm(xtr_norm, xts_norm, features)
    return xtr2_norm, xts2_norm



def local_iqr_global_norm(x_train, x_test, features):
    xtr_norm, xts_norm = local_iqr_norm(x_train, x_test, features)
    xtr2_norm, xts2_norm = global_full_norm(xtr_norm, xts_norm)
    return xtr2_norm, xts2_norm


def iqr_global_norm(x_train, x_test, features):
    xtr_norm, xts_norm = iqr_norm(x_train, x_test, features)
    xtr2_norm, xts2_norm = global_full_norm(xtr_norm, xts_norm)
    return xtr2_norm, xts2_norm

def make_norms_x(config, x_train, x_test, features):
    norm_type = config['features']['norm']
    if norm_type == '':
        return x_train, x_test
    if norm_type == 'global':
        return global_full_norm(x_train, x_test)
    elif norm_type == 'column':
        return global_column_wise_norm(x_train, x_test, features, return_bounds=False)
    elif norm_type == 'iqr':
        return iqr_norm(x_train, x_test, features)
    elif norm_type == 'iqr_global':
        return iqr_global_norm(x_train, x_test, features)
    elif norm_type == 'local_iqr':
        return local_iqr_norm(x_train, x_test, features)
    elif norm_type == 'local_iqr_global':
        return local_iqr_global_norm(x_train, x_test, features)
    elif norm_type == 'case_iqr_global':
        return case_iqr_global_norm(x_train, x_test, features)
    elif norm_type == 'global_max_log':
        return global_max_log_norm(x_train, x_test, features, log_type="log")
    elif norm_type == 'global_max_ln':
        return global_max_log_norm(x_train, x_test, features, log_type="ln")
    elif norm_type == 'local_max_log':
        return local_max_log_norm(x_train, x_test, features, log_type="log")
    elif norm_type == 'local_max_ln':
        return local_max_log_norm(x_train, x_test, features, log_type="ln")
    else:
        raise ValueError('[ERROR] No config[feat][norm] type was entered')


import numpy as np
from train_utils.calc_data_phys_const_losses import extract_rst

import numpy as np

def get_frob_scalar(config, y_train):
    output_feats = config['features']['output']
    max_frobs = []

    # Names of diagonal components
    diag_feats = {'uu', 'vv', 'ww'}

    for df in y_train.values():
        # Convert to NumPy array assuming order matches output_feats
        rst = df.values  # Shape: (N, 6)

        frob_sq = np.zeros(rst.shape[0])
        for i, feat in enumerate(output_feats):
            contrib = rst[:, i] ** 2
            if feat not in diag_feats:
                contrib *= 2.0  # Double off-diagonal components
            frob_sq += contrib

        frob = np.sqrt(frob_sq)
        max_frobs.append(np.max(frob))

    max_frob = np.abs(np.max(max_frobs))

    if max_frob == 0:
        raise ValueError("[ERROR] Max Frobenius norm is zero, can't normalize.")

    return max_frob


def make_norms_y(config, y_train, y_test):
    """
    Normalize Y with two independent scalers:
      - frob_max for the 6 tensor columns (RST6 or AIJ6)
      - k_max  for k/tke (if present)
    Returns (y_train_normed, y_test_normed, frob_max, k_max)
    """
    if config['features'].get('dnd', '') == 'nondim':
        # Skip normalization for nondimensional outputs
        return y_train, y_test, 1.0, 1.0

    output_feats = config['features']['output']
    six_names = _six_names_for_family(output_feats)

    low = [f.lower() for f in output_feats]
    has_k = ('tke' in low) or ('k' in low)
    k_name = output_feats[low.index('tke')] if 'tke' in low else (output_feats[low.index('k')] if 'k' in low else None)

    frob_max = get_frob_scalar(config, y_train)
    k_max   = get_k_max(config, y_train) if has_k else None

    y_train_normed, y_test_normed = {}, {}
    for split_name, ydict in [('train', y_train), ('test', y_test)]:
        out = {}
        for case, df in ydict.items():
            df2 = df.copy()
            present_six = [nm for nm in six_names if nm in df2.columns]
            if present_six:
                df2[present_six] = df2[present_six] / frob_max
            if has_k and k_name in df2.columns and k_max is not None:
                df2[k_name] = df2[k_name] / k_max
            out[case] = df2
        if split_name == 'train':
            y_train_normed = out
        else:
            y_test_normed = out

    return y_train_normed, y_test_normed, frob_max, k_max
