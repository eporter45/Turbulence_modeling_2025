# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 09:40:02 2025

@author: eoporter
Preprocessing pipeline for loading, filtering, normalizing, and visualizing 
shear mixing dataset cases. This script:
- Loads RANS and experimental data using a case trial system
- Filters features based on user-defined input/output selections
- Normalizes input features using a specified scheme
- Plots KDEs (Kernel Density Estimates) for visual inspection
- Converts data into PyTorch tensors for model training

Required configuration keys (passed via `cfg`) control data paths, feature setup,
trial selection, normalization, and more.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import pandas as pd
import matplotlib.pyplot as plt

#Dynamically resolve project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add to system path if not already
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def get_trial_cases(cfg, trials):
    trial = cfg['trial_name']
    print(f'Trial name: {trial}')
    test_cases = trials[trial]['test']
    train_cases = trials[trial]['train']
    return train_cases, test_cases

dataset_dir = os.path.join(PROJECT_ROOT, 'Data', 'Shear_mixing')
exp_dir = os.path.join(dataset_dir, 'train_exp')
rans_dir = os.path.join(dataset_dir, 'RANS', 'training')

def load_dfs(cfg, trials,  ranspath='', exppath = ''):
    train_cases, test_cases = get_trial_cases(cfg, trials)
    x_train, y_train = {}, {}
    x_test, y_test = {}, {}
    grid_dict = {'train': {}, 'test': {}}
    dnd = cfg['features']['dnd']
    gt = cfg['features']['grad_type']
    exp_data_dir = os.path.join(exppath, dnd)
    rans_data_dir = os.path.join(ranspath, gt)
    grid_feats = ['Cx', 'Cy']
    for train_case in train_cases:
        #get case parts
        case = train_case.split('_')[0]
        fov = train_case.split('_')[1]
        #get loading info
        rans_filename = f'{case}_{dnd}_{fov}.pkl'
        exp_filename = f'{dnd}_{case}_{fov}.pkl'
        rans = pd.read_pickle(os.path.join(rans_data_dir, rans_filename))
        #print(f'RANS KEYS: {rans.keys()}')
        exp = pd.read_pickle(os.path.join(exp_data_dir, exp_filename))
        #print(f'[INFO] exp keys {exp.columns}')
        #get features
        x_train[train_case] = rans
        y_train[train_case] = exp
    for test_case in test_cases:
        case = test_case.split('_')[0]
        fov = test_case.split('_')[1]
        rans_filename = f'{case}_{dnd}_{fov}.pkl'
        exp_filename = f'{dnd}_{case}_{fov}.pkl'
        rans = pd.read_pickle(os.path.join(rans_data_dir, rans_filename))
        exp = pd.read_pickle(os.path.join(exp_data_dir, exp_filename))
        x_test[test_case] = rans
        y_test[test_case] = exp
    return {'x_train': x_train, 'x_test': x_test,
            'y_train': y_train, 'y_test': y_test, }

from Plotting.plot_lumley import _exp_centerline_c123, _rans_centerline_c123

def filter_features(cfg, data_bundle):
    from Features.make_featuresets import get_feature_set
    feats = cfg['features']
    x_train, x_test = data_bundle['x_train'], data_bundle['x_test']
    y_train, y_test = data_bundle['y_train'], data_bundle['y_test']
    lumley_dict = {'train': {'RANS': {}, 'EXP': {}}, 'test': {'RANS': {}, 'EXP': {}}}
    for name, df in x_train.items():
        lumley_dict['train']['RANS'][name] = _rans_centerline_c123(df, n=20, tol_y=1.5e-3, x_col='Cx', y_col='Cy', eps=1e-9) 
    for name, df in x_test.items():
       lumley_dict['test']['RANS'][name] = _rans_centerline_c123(df, n=20, tol_y=1.5e-3, eps=1e-9, x_col='Cx', y_col='Cy')
    for name, df in y_train.items():
       lumley_dict['train']['EXP'][name] = _exp_centerline_c123(df, n=20, tol_y=1.5e-3, eps=1e-9, x_col='Cx', y_col='Cy')
    for name, df in y_test.items():
        lumley_dict['test']['EXP'][name] = _exp_centerline_c123(df, n=20, tol_y=1.5e-3, eps=1e-9, x_col='Cx', y_col='Cy')
        
       
    if not feats['in_is_out']:
        input_feats = get_feature_set(cfg)
        output_feats = feats['output']

        # --- Standard case ---
        x_train = {k: df[input_feats] for k, df in x_train.items()}
        x_test = {k: df[input_feats] for k, df in x_test.items()}
        y_train = {k: df[output_feats] for k, df in y_train.items()}
        y_test = {k: df[output_feats] for k, df in y_test.items()}
        return {'x_train': x_train, 'x_test': x_test,
                'y_train': y_train, 'y_test': y_test}, lumley_dict, input_feats
    else:
        input_feats = feats['input']
        output_feats = feats['output']
        if input_feats != output_feats:
            raise ValueError('[Error] in/out feats dont match')
        # --- Input is output: determine feature source ---
        if set(x_train.keys()) != set(y_train.keys()) or set(x_test.keys()) != set(y_test.keys()):
            raise ValueError('[ERROR] Train/test case keys are not matching between RANS and EXP')
        # Assume all cases have same structure
        sample_case = list(x_train.keys())[0]

        in_from_exp = all(feat in y_train[sample_case].columns for feat in input_feats)
        in_from_rans = all(feat in x_train[sample_case].columns for feat in input_feats)

        if not (in_from_exp or in_from_rans):
            raise ValueError("[ERROR] Input features not found in either EXP or RANS data.")

        # Select source dict for inputs
        input_source_train = y_train if in_from_exp else x_train
        input_source_test = y_test if in_from_exp else x_test

        # Build filtered dicts
        x_train = {k: input_source_train[k][input_feats] for k in input_source_train}
        x_test = {k: input_source_test[k][input_feats] for k in input_source_test}
        y_train = {k: y_train[k][output_feats] for k in y_train}
        y_test = {k: y_test[k][output_feats] for k in y_test}

        return {'x_train': x_train, 'x_test': x_test,
                'y_train': y_train, 'y_test': y_test},lumley_dict, input_feats
        
    
    

def stack_data(data_bundle):
    xtr = data_bundle['x_train']
    xts = data_bundle['x_test']
    ytr = data_bundle['y_train']
    yts = data_bundle['y_test']
    x_train, x_test = [], []
    y_train, y_test = [], []
    for xt in xtr:
        x_train.append(xtr[xt])
    for xs in xts:
        x_test.append(xts[xs])
    for yt in ytr:
        y_train.append(ytr[yt])
    for ys in yts:
        y_test.append(yts[ys])
    return {'x_train': x_train, 'x_test': x_test,
            'y_train': y_train, 'y_test': y_test}


def load_grid_dicts(data_bundle):
    grid_dict = {'test': {}, 'train': {}}
    test_keys = list(data_bundle['x_test'].keys())
    train_keys = list(data_bundle['x_train'].keys())
    grid_feats = ['Cx', 'Cy']
    k = ''
    for key in test_keys:
        grid_dict['test'][key] = (data_bundle['x_test'][key]['Cx'], data_bundle['x_test'][key]['Cy'])
    for key in train_keys:
        grid_dict['train'][key] = (data_bundle['x_train'][key]['Cx'], data_bundle['x_train'][key]['Cy'])
        k = key
    print(f'[DEBUG] grid_dict keys: {list(grid_dict.keys())} ')
    print(f"[DEBUG] grid_dict train type: {len(grid_dict['train'][k])} ")

    return grid_dict



def check_nan_outputs(data_bundle):
    for split in ['y_train', 'y_test']:
        for case, df in data_bundle[split].items():
            if df.isnull().values.any():
                print(f"[WARNING] NaNs found in {split} for case {case}")
                print(df[df.isnull().any(axis=1)])
            else:
                print(f'[INFO], no nans present in {split}, {case}')

from PreProcess.normalize_data import make_norms_x, make_norms_y
import copy
def load_and_norm(cfg, exp_dir, rans_dir):
    from Trials import TRIALS
    
    raw_data_bundle = load_dfs(cfg, TRIALS, rans_dir, exp_dir)   
    data_bundle,lumley_dict, x_feats = filter_features(cfg, raw_data_bundle)
    print(f'[INFO] Features Used: {x_feats}')
    check_nan_outputs(data_bundle)
    data_bundle['grid_dict'] = load_grid_dicts(raw_data_bundle)

    x_train_normed, x_test_normed = make_norms_x(cfg, data_bundle['x_train'],
                                              data_bundle['x_test'], x_feats)
    data_bundle['x_train_normed'] = x_train_normed
    data_bundle['x_test_normed'] = x_test_normed
    y_train_normed, y_test_normed = copy.deepcopy(data_bundle['y_train']), copy.deepcopy(data_bundle['y_test'])
    if cfg['features']['y_norm']:
        y_train_normed, y_test_normed, max_frob, max_k = make_norms_y(cfg, data_bundle['y_train'], data_bundle['y_test'])
        data_bundle['y_max_frob'] = max_frob
        data_bundle['y_max_k'] = max_k

    data_bundle['y_train_normed'] = y_train_normed
    data_bundle['y_test_normed'] = y_test_normed
    data_bundle['lumley_dict'] = lumley_dict
    return data_bundle





import seaborn as sns

def plot_kdes_by_feature(x_train, x_test, norm='raw', save=False, save_path=''):
    all_data = {**x_train, **x_test}
    feature_names = list(next(iter(all_data.values())).columns)

    for feat in feature_names:
        plt.figure(figsize=(8, 4))
        plotted = False
        for case, df in all_data.items():
            if df[feat].nunique() > 1:
                sns.kdeplot(df[feat], label=case, linewidth=1.5)
                plotted = True
            else:
                print(f"[INFO] Skipping KDE for feature {feat} in case {case} due to 0 variance")

        if plotted:
            plt.title(f'KDE for {norm} Feature: {feat}')
            plt.xlabel('Normalized Value')
            plt.ylabel('Density')
            plt.legend()
            plt.tight_layout()

            if save:
                if save_path == '':
                    raise ValueError('[Error] no savepath passed into function')
                fname = f'{feat}_{norm}_case_kde.png'
                fpath = os.path.join(save_path, fname)
                plt.savefig(fpath)
                plt.close()
            else:
                plt.show()
        else:
            plt.close()  # nothing plotted, avoid showing blank plots



def plot_kdes_by_case(x_train, x_test, norm='raw', save=False, save_path=''):
    all_data = {**x_train, **x_test}
    
    for case, df in all_data.items():
        for feat in df.columns:
            if df[feat].nunique() > 1:
                plt.figure(figsize=(10, 6))
                sns.kdeplot(df[feat], label=feat, linewidth=1.5)
                plt.title(f'KDE for {norm} Case: {case} - Feature: {feat}')
                plt.xlabel('Normalized Value')
                plt.ylabel('Density')
                plt.legend()
                plt.tight_layout()

                if save:
                    if save_path == '':
                        raise ValueError('[Error] no savepath passed into function')
                    fname = f'{case}_{feat}_{norm}_kde.png'
                    fpath = os.path.join(save_path, fname)
                    plt.savefig(fpath)
                    plt.close()
                else:
                    plt.show()
            else:
                print(f"[INFO] Skipping KDE for {feat} in case {case} due to 0 variance")

import torch

def convert_data_bundle_to_tensors(data_bundle, config):
    def convert_dict_to_tensor_list(d):
        return [torch.tensor(df.values, dtype=torch.float32) for df in d.values()]
    tensor_bundle = {
        'x_train': convert_dict_to_tensor_list(data_bundle['x_train']),
        'x_test': convert_dict_to_tensor_list(data_bundle['x_test']),
        'y_train': convert_dict_to_tensor_list(data_bundle['y_train']),
        'y_test': convert_dict_to_tensor_list(data_bundle['y_test']),
        'x_train_normed': convert_dict_to_tensor_list(data_bundle['x_train_normed']),
        'x_test_normed': convert_dict_to_tensor_list(data_bundle['x_test_normed']),
        'y_train_normed': convert_dict_to_tensor_list(data_bundle['y_train_normed']),
        'y_test_normed': convert_dict_to_tensor_list(data_bundle['y_test_normed']),
    }
    if config['features']['y_norm']:
        tensor_bundle['y_max_frob'] = data_bundle['y_max_frob']
        tensor_bundle['y_max_k'] = data_bundle['y_max_k']

    # Pass through grid_dict if present
    if 'grid_dict' in data_bundle:
        tensor_bundle['grid_dict'] = data_bundle['grid_dict']
        print('[INFO] grid dict added to tensor bundle')
    if 'lumley_dict' in data_bundle:
        tensor_bundle['lumley_dict'] = data_bundle['lumley_dict']
        print(f'[INFO] lumley dict added to tensor bundle')
    return tensor_bundle


def load_norm_kde(cfg, exp_dir, rans_dir, save=False, save_path=''):
    # Load + normalize
    data_bundle = load_and_norm(cfg, exp_dir, rans_dir)
    
    # Get save path (if enabled)
    savepath = cfg['paths']['output_dir'] if save else ''

    # Determine label for normalized plots
    norm_label = cfg['features']['norm'] + ' norm'

    # === Plot KDEs by feature: all cases shown together, one plot per feature ===
    print("Plotting KDEs by feature across all cases...")
    plot_kdes_by_feature(data_bundle['x_train_normed'], data_bundle['x_test_normed'],
                         norm=norm_label, save=save, save_path=savepath)

    # === Plot KDEs by case: all features in each case, one plot per case ===
    #print("Plotting KDEs by case across all features...")
    #plot_kdes_by_case(data_bundle['x_train_normed'], data_bundle['x_test_normed'],
     #                 norm=norm_label, save=save_kde, save_path=savepath)

    # Convert to tensor and return
    tens_bundle = convert_data_bundle_to_tensors(data_bundle, config=cfg)
    return tens_bundle

     


#cfg for testing
'''
dataset_dir = os.path.join(PROJECT_ROOT, 'Data', 'Shear_mixing')
exp_dir = os.path.join(dataset_dir, 'train_exp')
rans_dir = os.path.join(dataset_dir, 'RANS', 'training')

config = {
    'debug': False,
    'input_is_output': True,
    "trial_name": "single_case",
    "features": {
        "dnd": 'nondim',
        'grad_type': 'MLS',
        "input": 'FS1',
        'norm': 'iqr_global',
        "output": ['uu', 'uv', 'vv', 'uw', 'vw', 'ww'],
        'trim_z': True,
    },
 }
load_norm_kde(config, exp_dir, rans_dir, False)
'''