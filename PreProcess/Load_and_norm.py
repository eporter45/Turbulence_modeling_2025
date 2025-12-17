# -*- coding: utf-8 -*-
"""
Preprocessing pipeline for loading, filtering, normalizing, and visualizing
shear-mixing datasets.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
#  Resolve project root
# ---------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------
#  Trial system helper
# ---------------------------------------------------------------
def get_trial_cases(cfg, trials):
    trial = cfg['trial_name']
    if trial not in trials:
        raise KeyError(f"[ERROR] Trial '{trial}' not found in TRIALS dict.")
    print(f"[INFO] Trial name: {trial}")

    train_cases = trials[trial].get('train', [])
    test_cases = trials[trial].get('test', [])

    if not train_cases:
        raise ValueError("[ERROR] No training cases found for this trial.")
    if not test_cases:
        raise ValueError("[ERROR] No testing cases found for this trial.")
    return train_cases, test_cases


# ---------------------------------------------------------------
#  Default Paths
# ---------------------------------------------------------------
dataset_dir = os.path.join(PROJECT_ROOT, 'Data', 'Shear_mixing')
exp_dir_default = os.path.join(dataset_dir, 'train_exp')
rans_dir_default = os.path.join(dataset_dir, 'RANS', 'training')


# ---------------------------------------------------------------
#  SAFE load_dfs()
# ---------------------------------------------------------------
def load_dfs(cfg, trials, ranspath='', exppath=''):
    """
    Load RANS and EXP data for selected cases.
    Includes path validation + missing-column checks.
    """

    # ---- SAFETY: paths required ----
    if not ranspath:
        raise ValueError("[ERROR] rans_dir not provided to load_dfs.")
    if not exppath:
        raise ValueError("[ERROR] exp_dir not provided to load_dfs.")

    if not os.path.isdir(ranspath):
        raise FileNotFoundError(f"[ERROR] RANS directory does not exist: {ranspath}")
    if not os.path.isdir(exppath):
        raise FileNotFoundError(f"[ERROR] EXP directory does not exist: {exppath}")

    train_cases, test_cases = get_trial_cases(cfg, trials)

    x_train, y_train = {}, {}
    x_test, y_test = {}, {}

    dnd = cfg['features']['dnd']
    gt = cfg['features']['grad_type']

    exp_data_dir = os.path.join(exppath, dnd)
    rans_data_dir = os.path.join(ranspath, gt)

    if not os.path.isdir(exp_data_dir):
        raise FileNotFoundError(f"[ERROR] EXP subdir missing: {exp_data_dir}")
    if not os.path.isdir(rans_data_dir):
        raise FileNotFoundError(f"[ERROR] RANS subdir missing: {rans_data_dir}")

    # -----------------------------------------------------------
    # Load TRAIN cases
    # -----------------------------------------------------------
    for cname in train_cases:
        case, fov = cname.split('_')
        rfile = os.path.join(rans_data_dir, f"{case}_{dnd}_{fov}.pkl")
        efile = os.path.join(exp_data_dir, f"{dnd}_{case}_{fov}.pkl")

        if not os.path.exists(rfile):
            raise FileNotFoundError(f"[ERROR] Missing RANS file: {rfile}")
        if not os.path.exists(efile):
            raise FileNotFoundError(f"[ERROR] Missing EXP file: {efile}")

        rans = pd.read_pickle(rfile)
        exp = pd.read_pickle(efile)

        # ---- SAFETY: grid columns must exist ----
        for col in ["Cx", "Cy"]:
            if col not in rans.columns:
                raise KeyError(f"[ERROR] RANS file missing grid column '{col}': {rfile}")
            if col not in exp.columns:
                raise KeyError(f"[ERROR] EXP file missing grid column '{col}': {efile}")

        x_train[cname] = rans
        y_train[cname] = exp

    # -----------------------------------------------------------
    # Load TEST cases (same checks)
    # -----------------------------------------------------------
    for cname in test_cases:
        case, fov = cname.split('_')
        rfile = os.path.join(rans_data_dir, f"{case}_{dnd}_{fov}.pkl")
        efile = os.path.join(exp_data_dir, f"{dnd}_{case}_{fov}.pkl")

        if not os.path.exists(rfile):
            raise FileNotFoundError(f"[ERROR] Missing RANS file: {rfile}")
        if not os.path.exists(efile):
            raise FileNotFoundError(f"[ERROR] Missing EXP file: {efile}")

        rans = pd.read_pickle(rfile)
        exp = pd.read_pickle(efile)

        for col in ["Cx", "Cy"]:
            if col not in rans.columns:
                raise KeyError(f"[ERROR] RANS file missing grid column '{col}': {rfile}")
            if col not in exp.columns:
                raise KeyError(f"[ERROR] EXP file missing grid column '{col}': {efile}")

        x_test[cname] = rans
        y_test[cname] = exp

    return {
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test,
    }


# ---------------------------------------------------------------
#  check_nan_outputs()
# ---------------------------------------------------------------
def check_nan_outputs(data_bundle):
    for key in ['y_train', 'y_test']:
        for case, df in data_bundle[key].items():
            if df.isnull().values.any():
                raise ValueError(f"[ERROR] NaNs found in {key} for case {case}")
            else:
                print(f"[INFO] No NaNs in {key} for case {case}")


# ---------------------------------------------------------------
#  load_grid_dicts
# ---------------------------------------------------------------
def load_grid_dicts(data_bundle):
    grid_dict = {'train': {}, 'test': {}}

    for split in ['train', 'test']:
        for case, df in data_bundle[f'x_{split}'].items():
            if "Cx" not in df.columns or "Cy" not in df.columns:
                raise KeyError(f"[ERROR] Missing Cx/Cy in data for {case}")
            grid_dict[split][case] = (df["Cx"], df["Cy"])

    return grid_dict


# ---------------------------------------------------------------
#  load_and_norm wrapper (unchanged logic)
# ---------------------------------------------------------------
from PreProcess.normalize_data import make_norms_x, make_norms_y
from Plotting.plot_lumley import _exp_centerline_c123, _rans_centerline_c123
from Features.make_featuresets import get_feature_set
import copy


def filter_features(cfg, data_bundle):
    """
    Injects Lumley dict + filters RANS/EXP columns.
    Safety checks added.
    """
    feats = cfg['features']
    x_train, x_test = data_bundle['x_train'], data_bundle['x_test']
    y_train, y_test = data_bundle['y_train'], data_bundle['y_test']

    # --- Lumley centerlines (safe now)
    lumley_dict = {'train': {'RANS': {}, 'EXP': {}},
                   'test': {'RANS': {}, 'EXP': {}}}

    # Check grid cols exist before Lumley
    for k, df in {**x_train, **y_train, **x_test, **y_test}.items():
        for c in ['Cx', 'Cy']:
            if c not in df.columns:
                raise KeyError(f"[ERROR] Missing {c} for Lumley centerline: {k}")

    for k, df in x_train.items():
        lumley_dict['train']['RANS'][k] = _rans_centerline_c123(df)
    for k, df in x_test.items():
        lumley_dict['test']['RANS'][k] = _rans_centerline_c123(df)
    for k, df in y_train.items():
        lumley_dict['train']['EXP'][k] = _exp_centerline_c123(df)
    for k, df in y_test.items():
        lumley_dict['test']['EXP'][k] = _exp_centerline_c123(df)

    # --- Filter features
    if not feats['in_is_out']:
        input_feats = get_feature_set(cfg)
        output_feats = feats['output']

        for feat in input_feats:
            for d in (x_train, x_test):
                if feat not in next(iter(d.values())).columns:
                    raise KeyError(f"[ERROR] Missing input feature '{feat}' in dataset.")

        x_train_f = {k: df[input_feats].copy() for k, df in x_train.items()}
        x_test_f = {k: df[input_feats].copy() for k, df in x_test.items()}
        y_train_f = {k: df[output_feats].copy() for k, df in y_train.items()}
        y_test_f = {k: df[output_feats].copy() for k, df in y_test.items()}

    else:
        # in_is_out mode
        input_feats = feats['input']
        output_feats = feats['output']
        try:
            x_train_f = {k: y_train[k][input_feats].copy() for k in y_train}
            x_test_f = {k: y_test[k][input_feats].copy() for k in y_test}
        except KeyError as e:
            raise KeyError(f"[ERROR] Feature mismatch in in_is_out mode: {e}")

        y_train_f = {k: y_train[k][output_feats].copy() for k in y_train}
        y_test_f = {k: y_test[k][output_feats].copy() for k in y_test}

    return {
        'x_train': x_train_f, 'x_test': x_test_f,
        'y_train': y_train_f, 'y_test': y_test_f
    }, lumley_dict, input_feats


# ---------------------------------------------------------------
#  Main load+norm wrapper
# ---------------------------------------------------------------
def load_and_norm(cfg, exp_dir, rans_dir):
    from Trials import TRIALS

    raw = load_dfs(cfg, TRIALS, rans_dir, exp_dir)

    data_bundle, lumley_dict, x_feats = filter_features(cfg, raw)
    print(f"[INFO] Features Used: {x_feats}")

    check_nan_outputs(data_bundle)
    data_bundle['grid_dict'] = load_grid_dicts(raw)

    # ---- Normalize X ----
    x_trn_norm, x_tst_norm = make_norms_x(cfg, data_bundle['x_train'],
                                          data_bundle['x_test'], x_feats)
    data_bundle['x_train_normed'] = x_trn_norm
    data_bundle['x_test_normed'] = x_tst_norm

    # ---- Normalize Y ----
    y_train_norm, y_test_norm = copy.deepcopy(data_bundle['y_train']), copy.deepcopy(data_bundle['y_test'])
    if cfg['features']['y_norm']:
        y_train_norm, y_test_norm, mf, mk = make_norms_y(cfg, data_bundle['y_train'], data_bundle['y_test'])
        data_bundle['y_max_frob'] = mf
        data_bundle['y_max_k'] = mk

    data_bundle['y_train_normed'] = y_train_norm
    data_bundle['y_test_normed'] = y_test_norm
    data_bundle['lumley_dict'] = lumley_dict

    return data_bundle
