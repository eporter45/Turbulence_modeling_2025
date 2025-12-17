# -*- coding: utf-8 -*-
"""
load_and_norm.py
Wrapper script for loading, filtering, normalizing, and visualizing shear-mixing datasets.

Delegates detailed logic to:
 - load_functions.py  → for loading/filtering
 - normalize_data.py  → for normalization
 - plotting_kde.py    → for visualization
"""

def load_norm(cfg, exp_dir, rans_dir, save=False, save_path=''):
    from Trials import TRIALS
    from PreProcess.Load_data import (
        load_dfs, filter_features, check_nan_outputs,
        load_grid_dicts, stack_data
    )
    from PreProcess.normalize_data import make_norms_x, make_norms_y
    from Plotting.Plot_kde import plot_kdes_by_feature
    import copy

    # === Step 1: Load raw data ===
    raw_data_bundle = load_dfs(cfg, TRIALS, rans_dir, exp_dir)

    # === Step 2: Model-specific branch ===
    model_type = cfg['model']['type'].lower()
    if model_type == 'tbnn':
        from PreProcess.get_tbnn_features import load_tbnn_data
        tbnn_bundle = load_tbnn_data(cfg, raw_data_bundle)
        print("[INFO] TBNN preprocessing complete.")
        print(f"[INFO] Invariants used: {cfg['features']['invariants']}")
        print(f"[INFO] Tensor basis used: {cfg['features']['tensor_basis']}")
        return tbnn_bundle

    # === Step 3: Feature filtering + Lumley dictionary ===
    data_bundle, lumley_dict, x_feats = filter_features(cfg, raw_data_bundle)
    print(f'[INFO] Features Used: {x_feats}')

    # === Step 4: Check NaNs, attach grids ===
    check_nan_outputs(data_bundle)
    data_bundle['grid_dict'] = load_grid_dicts(raw_data_bundle)

    # === Step 5: Normalize X ===
    x_train_normed, x_test_normed = make_norms_x(
        cfg, data_bundle['x_train'], data_bundle['x_test'], x_feats
    )
    data_bundle['x_train_normed'] = x_train_normed
    data_bundle['x_test_normed'] = x_test_normed

    # === Step 5b: Normalize Y ===
    y_train_normed, y_test_normed = data_bundle['y_train'], data_bundle['y_test']
    if cfg['features']['y_norm']:
        y_train_normed, y_test_normed, max_frob, max_k = make_norms_y(
            cfg, data_bundle['y_train'], data_bundle['y_test']
        )
        data_bundle['y_max_frob'] = max_frob
        data_bundle['y_max_k'] = max_k

    data_bundle['y_train_normed'] = y_train_normed
    data_bundle['y_test_normed'] = y_test_normed
    data_bundle['lumley_dict'] = lumley_dict

    # === Step 6: KDE visualization ===
    if save:
        norm_label = cfg['features']['norm'] + ' norm'
        print("Plotting KDEs by feature across all cases...")
        plot_kdes_by_feature(
            x_train_normed, x_test_normed,
            norm=norm_label, save=save, save_path=save_path
        )

    return data_bundle


