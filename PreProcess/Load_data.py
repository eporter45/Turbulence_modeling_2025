"""
load_functions.py
Handles loading, filtering, and basic sanity checks for RANS and experimental data.
"""

import pandas as pd
from Plotting.plot_lumley import _exp_centerline_c123, _rans_centerline_c123
import os

def load_data(data_root, case_list, fov_list=None, mode='nondim'):
    """
    Load data for TBNN training, which requires both RANS and EXP feature fields
    aligned per case and FOV.

    Parameters
    ----------
    data_root : str
        Base path to 'Training_data_10_2025/'.
    case_list : list of str
        Case names (e.g. ['Case1', 'Case2']).
    fov_list : list of str, optional
        Specific FOVs to load (e.g. ['FOV1','FOV2']). Loads all if None.
    mode : str
        'dim' or 'nondim' (default: 'nondim').

    Returns
    -------
    tbnn_data : dict
        {
            'Case1': {
                'FOV1': {'rans': df_rans, 'exp': df_exp},
                'FOV2': {...}
            },
            ...
        }
    """
    import os
    import pandas as pd

    dfs_data = {}
    for case in case_list:
        case_dir = os.path.join(data_root, case)
        dfs_data[case] = {}

        if fov_list is None:
            # auto-detect FOVs from filenames
            fovs = sorted({f.split('_')[2] for f in os.listdir(case_dir) if f.endswith('.pkl')})
        else:
            fovs = fov_list

        for fov in fovs:
            key = f"{case}_{fov}"
            rans_file = os.path.join(case_dir, f"{key}_rans_{mode}.pkl")
            exp_file  = os.path.join(case_dir, f"{key}_exp_{mode}.pkl")

            if os.path.exists(rans_file) and os.path.exists(exp_file):
                df_rans = pd.read_pickle(rans_file)
                df_exp  = pd.read_pickle(exp_file)
                dfs_data[case][fov] = {'rans': df_rans.copy(), 'exp': df_exp.copy()}
            else:
                print(f"[WARN] Missing data for {key}: {rans_file if not os.path.exists(rans_file) else exp_file}")

    return dfs_data


def get_trial_cases(cfg, trials):
    trial = cfg['trial_name']
    print(f'Trial name: {trial}')
    return trials[trial]['train'], trials[trial]['test']

def load_dfs(cfg, trials, ranspath='', exppath=''):
    train_cases, test_cases = get_trial_cases(cfg, trials)
    x_train, y_train, x_test, y_test = {}, {}, {}, {}
    dnd, gt = cfg['features']['dnd'], cfg['features']['grad_type']

    for split, cases, X, Y in [('train', train_cases, x_train, y_train),
                               ('test', test_cases, x_test, y_test)]:
        for case_name in cases:
            case, fov = case_name.split('_')[0], case_name.split('_')[1]
            rans_file = os.path.join(ranspath, case, f"{case}_chkpt6_train_{fov}_rans_{dnd}.pkl")
            exp_file = os.path.join(ranspath, case, f"{case}_chkpt6_train_{fov}_exp_{dnd}.pkl")

            X[case_name] = pd.read_pickle(rans_file)
            Y[case_name] = pd.read_pickle(exp_file)

    return {'x_train': x_train, 'x_test': x_test,
            'y_train': y_train, 'y_test': y_test}

def filter_features(cfg, data_bundle):
    from Features.make_featuresets import get_feature_set
    from utils.io_tools import detect_output_cols
    feats = cfg['features']
    output_fam = feats['output_family']
    out_feats = detect_output_cols(output_fam)
    x_train, x_test, y_train, y_test = (
        data_bundle['x_train'], data_bundle['x_test'],
        data_bundle['y_train'], data_bundle['y_test']
    )

    lumley_dict = {'train': {'RANS': {}, 'EXP': {}}, 'test': {'RANS': {}, 'EXP': {}}}
    for name, df in x_train.items():
        lumley_dict['train']['RANS'][name] = _rans_centerline_c123(df, n=20, tol_y=1.5e-3)
    for name, df in x_test.items():
        lumley_dict['test']['RANS'][name] = _rans_centerline_c123(df, n=20, tol_y=1.5e-3)
    for name, df in y_train.items():
        lumley_dict['train']['EXP'][name] = _exp_centerline_c123(df, n=20, tol_y=1.5e-3)
    for name, df in y_test.items():
        lumley_dict['test']['EXP'][name] = _exp_centerline_c123(df, n=20, tol_y=1.5e-3)

    if not feats['in_is_out']:
        input_feats = get_feature_set(cfg)
        output_feats = out_feats
        x_train = {k: df[input_feats].copy() for k, df in x_train.items()}
        x_test = {k: df[input_feats].copy() for k, df in x_test.items()}
        y_train = {k: df[output_feats].copy() for k, df in y_train.items()}
        y_test = {k: df[output_feats].copy() for k, df in y_test.items()}
    else:
        input_feats = feats['input']
        output_feats = out_feats
        x_train = {k: y_train[k][input_feats].copy() for k in y_train}
        x_test = {k: y_test[k][input_feats].copy() for k in y_test}
        y_train = {k: y_train[k][output_feats].copy() for k in y_train}
        y_test = {k: y_test[k][output_feats].copy() for k in y_test}

    return {'x_train': x_train, 'x_test': x_test,
            'y_train': y_train, 'y_test': y_test}, lumley_dict, input_feats



def check_nan_outputs(data_bundle):
    for split in ['y_train', 'y_test']:
        for case, df in data_bundle[split].items():
            if df.isnull().values.any():
                print(f"[WARN] NaNs in {split} for {case}")
            else:
                print(f"[INFO] {split} {case} clean")

def load_grid_dicts(data_bundle):
    grid_dict = {'train': {}, 'test': {}}
    for split in ['train', 'test']:
        for key, df in data_bundle[f'x_{split}'].items():
            grid_dict[split][key] = (df['Cx'], df['Cy'])
    return grid_dict

def stack_data(data_bundle):
    return {
        'x_train': list(data_bundle['x_train'].values()),
        'x_test': list(data_bundle['x_test'].values()),
        'y_train': list(data_bundle['y_train'].values()),
        'y_test': list(data_bundle['y_test'].values())
    }
