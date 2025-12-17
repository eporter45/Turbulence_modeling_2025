# -*- coding: utf-8 -*-
"""
Clean Normalization Script for ML Turbulence Pipeline
Last updated: 2025-11-17
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

# -------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------
RST6 = ['uu','uv','vv','uw','vw','ww']
AIJ6 = ['a_xx','a_xy','a_yy','a_xz','a_yz','a_zz']


# -------------------------------------------------------------
# FAMILY HELPERS
# -------------------------------------------------------------
def _detect_family(output_features):
    low = [c.lower() for c in output_features]
    if any(n.startswith('a_') for n in low): return 'aij'
    if any(n.startswith('b_') for n in low): return 'bij'
    return 'rst'


def _six_names_for_family(output_features):
    fam = _detect_family(output_features)
    if fam == 'aij':
        return [nm for nm in AIJ6 if nm in output_features]
    return [nm for nm in RST6 if nm in output_features]


def _diag_names_for_family(output_features):
    fam = _detect_family(output_features)
    if fam == 'aij':
        return ['a_xx','a_yy','a_zz']
    return ['uu','vv','ww']


# -------------------------------------------------------------
# Y-SCALAR HELPERS
# -------------------------------------------------------------
def get_k_max(config, y_train):
    feats = config['features']['output']
    low = [f.lower() for f in feats]

    if 'tke' not in low and 'k' not in low:
        return None

    k_name = feats[low.index('tke')] if 'tke' in low else feats[low.index('k')]
    vals = np.concatenate([df[k_name].values for df in y_train.values()])
    kmax = np.max(np.abs(vals))
    return max(kmax, 1e-12)


def get_frob_scalar(config, y_train):
    """Max Frobenius norm for 6-column tensor output."""
    feats = config['features']['output']
    diag = {'uu','vv','ww','a_xx','a_yy','a_zz'}

    max_vals = []
    for df in y_train.values():
        arr = df[feats].values  # assume 6-tensor columns
        frob_sq = np.zeros(arr.shape[0])
        for i, ft in enumerate(feats):
            contrib = arr[:, i] ** 2
            if ft not in diag:
                contrib *= 2.0
            frob_sq += contrib
        max_vals.append(np.sqrt(frob_sq).max())

    max_f = abs(max(max_vals))
    return max(max_f, 1e-12)


# -------------------------------------------------------------
# TAIL CLIPPING
# -------------------------------------------------------------
def is_bimodal(series, prominence=0.01):
    data = series.dropna().values
    if len(data) < 2:
        return False

    kde = gaussian_kde(data)
    xs = np.linspace(data.min(), data.max(), 6000)
    ys = kde(xs)
    peaks, _ = find_peaks(ys, prominence=prominence)
    return len(peaks) >= 2


def get_tail_clip_bounds(df, features, kurt_thresh=1.5):
    bounds = {}
    for feat in features:
        col = df[feat]
        if col.var() == 0:
            continue

        if is_bimodal(col):
            low_pct, high_pct = 0.01, 0.01
        else:
            if col.kurtosis() < kurt_thresh:
                low_pct, high_pct = 0.02, 0.05
            else:
                low_pct, high_pct = 0.02, 0.10

        lo = col.quantile(low_pct)
        hi = col.quantile(1 - high_pct)
        bounds[feat] = (lo, hi)

    return bounds


def apply_bounds(df, bounds):
    df2 = df.copy()
    for feat, (lo, hi) in bounds.items():
        df2[feat] = (df2[feat] - lo) / (hi - lo + 1e-12)
    return df2


# -------------------------------------------------------------
# NORMALIZATION METHODS
# -------------------------------------------------------------
def global_full_norm(x_train, x_test):
    """Normalize ALL features using global train min/max."""
    stacked = pd.concat([df for df in x_train.values()])
    gmin = stacked.min().min()
    gmax = stacked.max().max()

    xtr, xts = {}, {}
    for k, df in x_train.items():
        xtr[k] = (df - gmin) / (gmax - gmin + 1e-12)
    for k, df in x_test.items():
        xts[k] = (df - gmin) / (gmax - gmin + 1e-12)
    return xtr, xts


def global_column_wise_norm(x_train, x_test, features):
    stacked = pd.concat([df[features] for df in x_train.values()])
    bounds = get_tail_clip_bounds(stacked, features)

    xtr, xts = {}, {}
    for k, df in x_train.items():
        xtr[k] = apply_bounds(df, bounds)
    for k, df in x_test.items():
        xts[k] = apply_bounds(df, bounds)
    return xtr, xts


def global_max_log_norm(x_train, x_test, features, log_type="log"):
    stacked = pd.concat([df[features] for df in x_train.values()])
    bounds = get_tail_clip_bounds(stacked, features)
    max_dict = {f: bounds[f][1] for f in features}

    def tr(x, m):
        if log_type == "log":
            return np.log(np.abs(x) + 1e-8) / np.log(m + 1e-8) * np.sign(x)
        return np.log1p(np.abs(x) / (m + 1e-8)) * np.sign(x)

    xtr, xts = {}, {}
    for k, df in x_train.items():
        tmp = df.copy()
        for f in features:
            tmp[f] = tr(df[f], max_dict[f])
        xtr[k] = tmp

    for k, df in x_test.items():
        tmp = df.copy()
        for f in features:
            tmp[f] = tr(df[f], max_dict[f])
        xts[k] = tmp

    return xtr, xts


def iqr_norm(x_train, x_test, features):
    stacked = pd.concat([df[features] for df in x_train.values()])
    stats = {f: stacked[f].quantile(0.75) - stacked[f].quantile(0.25) for f in features}

    xtr, xts = {}, {}
    for k, df in x_train.items():
        tmp = df.copy()
        for f in features:
            tmp[f] = tmp[f] / (stats[f] + 1e-12)
        xtr[k] = tmp

    for k, df in x_test.items():
        tmp = df.copy()
        for f in features:
            tmp[f] = tmp[f] / (stats[f] + 1e-12)
        xts[k] = tmp

    return xtr, xts


def local_iqr_norm(x_train, x_test, features):
    xtr, xts = {}, {}

    for k, df in x_train.items():
        tmp = df.copy()
        stats = {f: df[f].quantile(0.75) - df[f].quantile(0.25) for f in features}
        for f in features:
            tmp[f] = tmp[f] / (stats[f] + 1e-12)
        xtr[k] = tmp

    for k, df in x_test.items():
        tmp = df.copy()
        stats = {f: df[f].quantile(0.75) - df[f].quantile(0.25) for f in features}
        for f in features:
            tmp[f] = tmp[f] / (stats[f] + 1e-12)
        xts[k] = tmp

    return xtr, xts
def local_max_log_norm(x_train, x_test, features, log_type="log"):
    """
    Local max-based log normalization.

    For each case:
        - Compute robust per-feature max using upper tail clip bound.
        - Apply log or ln normalization:
            log: log(|x| + eps) / log(max_val + eps) * sign(x)
            ln:  sign(x) * log1p(|x| / (max_val + eps))

    Args:
        x_train, x_test : dict of {case_name : DataFrame}
        features        : list of feature names to normalize
        log_type        : 'log' or 'ln'

    Returns:
        x_train_normed, x_test_normed
    """

    def transform(x, m, eps=1e-8):
        if log_type == "log":
            return np.log(np.abs(x) + eps) / np.log(m + eps) * np.sign(x)
        elif log_type == "ln":
            return np.log1p(np.abs(x) / (m + eps)) * np.sign(x)
        else:
            raise ValueError(f"[ERROR] Unknown log_type '{log_type}'")

    x_train_normed = {}
    x_test_normed = {}

    # ---------------------------
    # TRAIN: compute per-case max
    # ---------------------------
    for case, df in x_train.items():
        clip_bounds = get_tail_clip_bounds(df[features], features)
        max_dict = {feat: clip_bounds[feat][1] for feat in features}

        df_norm = df.copy()
        for feat in features:
            df_norm[feat] = transform(df[feat].values, max_dict[feat])
        x_train_normed[case] = df_norm

    # ---------------------------
    # TEST: compute per-case max
    # ---------------------------
    for case, df in x_test.items():
        clip_bounds = get_tail_clip_bounds(df[features], features)
        max_dict = {feat: clip_bounds[feat][1] for feat in features}

        df_norm = df.copy()
        for feat in features:
            df_norm[feat] = transform(df[feat].values, max_dict[feat])
        x_test_normed[case] = df_norm

    return x_train_normed, x_test_normed


# -------------------------------------------------------------
# DISPATCHERS
# -------------------------------------------------------------
def make_norms_x(config, x_train, x_test, features):
    mode = config['features']['norm']

    if mode == '' or mode is None:
        return x_train, x_test
    if mode == 'global':
        return global_full_norm(x_train, x_test)
    if mode == 'column':
        return global_column_wise_norm(x_train, x_test, features)
    if mode == 'iqr':
        return iqr_norm(x_train, x_test, features)
    if mode == 'local_iqr':
        return local_iqr_norm(x_train, x_test, features)
    if mode == 'global_max_log':
        return global_max_log_norm(x_train, x_test, features, log_type="log")
    if mode == 'global_max_ln':
        return global_max_log_norm(x_train, x_test, features, log_type="ln")
    if mode == 'local_max_log':
        return local_max_log_norm(x_train, x_test, features, log_type="log")
    if mode == 'local_max_ln':
        return local_max_log_norm(x_train, x_test, features, log_type="ln")

    raise ValueError(f"[ERROR] Unknown X normalization mode '{mode}'")


def make_norms_y(config, y_train, y_test):
    if config['features'].get('dnd', '') == 'nondim':
        return y_train, y_test, 1.0, 1.0

    feats = config['features']['output']
    six_names = _six_names_for_family(feats)

    frob_max = get_frob_scalar(config, y_train)
    k_max = get_k_max(config, y_train)

    ytr, yts = {}, {}
    for split, ydict in [('train', y_train), ('test', y_test)]:
        out = {}
        for case, df in ydict.items():
            tmp = df.copy()

            # 6-tensor scaling
            present = [nm for nm in six_names if nm in tmp.columns]
            if present:
                tmp[present] = tmp[present] / frob_max

            # TKE scaling
            if k_max is not None:
                low = [c.lower() for c in feats]
                k_name = feats[low.index('tke')] if 'tke' in low else feats[low.index('k')]
                if k_name in tmp.columns:
                    tmp[k_name] = tmp[k_name] / k_max

            out[case] = tmp

        if split == 'train':
            ytr = out
        else:
            yts = out

    return ytr, yts, frob_max, k_max
