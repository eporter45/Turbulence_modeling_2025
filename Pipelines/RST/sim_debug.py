# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 09:54:50 2025

@author: eoporter
"""

# debug_main.py
import torch
from pathlib import Path
import pandas as pd
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
from PreProcess.normalize_data import normalize_x_test_train
from Plotting.Plot_kde import plot_input_kde
from Models.FCN import runSimple_model
#from Models.KAN import runKAN_model
from Post_Process.compute_test_losses import compute_test_losses_and_save
from Post_Process.make_accuracy_metrics import evaluate_cases
from Post_Process.organize_results import organize_rst_results
from Plotting.Plot_preds import plot_all_tensor_fields
from Plotting.plot_train_preds import evaluate_training_cases
from Plotting.Plot_loss_histories import plot_all_losses
from train_utils.make_loss_weights import plot_weights_history
import sys
from Trials import debug_trials
#Dynamically resolve project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Add to system path if not already
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)
print(project_root)
# Define key paths
big_data_dir = os.path.join(project_root, 'data')
source_dir = os.path.join(big_data_dir, 'Shear_mixing')
exp_dir = os.path.join(source_dir, 'merged_exp')
data_path = os.path.join(exp_dir, 'Case1_FOV1_data.pkl')
# === Hardcoded Debug Config ===
config = {
    'debug': True,
    "trial_name": "debug1",
    "features": {
        "input": ["U"],
        "output": ["U"]
    },
    "paths": {
        "name": "debug_run",
        "output_dir": "Outputs/debug"
    },
    "model": {
        'type': 'FCN',                     
        'layers': 15  ,                     # Number of total layers (including input/output)
        'width': 10,                       # Width of each hidden layer
        'dropout': 0.1,                    # Dropout rate (used every other layer)
        'activation': 'leakyrelu',         # Options: 'relu', 'leakyrelu', 'tanh', 'sigmoid'
        'seed': 42                        # For deterministic behavior
    },
    "eval_training_cases": False,

    'training':{
      'epochs': 1000,
      'batch_size': 1024,
      'eval_every': 10 ,                 # How often to print evaluation output
      'seed': 42,
      'criterion': 'mse',
      'optimizer': 'adam',
      'lr': 0.0001,
      'lambda': 0.0,
      'alpha': 0.0,                     #for RMS propr optimizer
      'scheduler':{
         'enabled': True,
         'type': 'reduce',      # Options: 'reduce', 'step', 'cosine', etc.
         'factor': 0.5,
         'patience': 10,
         'cooldown': 10,
         'min_lr': 1e-6,
         'step_size': 50,                  # Used for StepLR
         'gamma': 0.9,                     # For StepLR or CosineAnnealing
         },
                       

      # ---- Loss Configuration ----
      'loss':{
        'weights':{
          'enabeled': False,
          'data': {
              'enabeled': False,
              'dynamic': False,
              },
          'group':{
            'enabeled': False,
            'dynamic': False,
            'data': 1.0,
            'phys': 1.0,
            'constraint': 1.0,
            },
          'case_weights': {
              'enabeled': False,
              'dynamic': False,
              'type': '',
              'reduction':'' , 
              },
          'phase_switch':{
            'enabled': False,
            'loss_key': "invariants",     # loss key to trigger phase switching
            'threshold_percent': 0.1,
            'pre_switch_weight': 0.01,
            'post_switch_weight': 1.0,
            },
          },
        'terms': {
            'phys':{
              'enabled': False,
              'dynamic': False,
              'types': ['tke', 'aij', 'bij'],
                },
            'constraint':{
              'enabled': False,
              'dynamic': False,
              'types':['invariant_supervision', 
                       'eigen_invariants',
                       'bij_aij_eig_relation'],
                        }, 
                    },           
                },
              },
            }

def load_data(data_dir, train_cases, test_cases, input_features, output_features):
    def load_case(case_name):
        case_file = os.path.join(data_dir, f"{case_name}.pkl")
        df = pd.read_pickle(case_file).dropna()
        
        # Extract input/output
        x_np = df[input_features].values.astype('float32')
        y_np = df[output_features].values.astype('float32')
        x_t = torch.from_numpy(x_np)
        y_t = torch.from_numpy(y_np)

        # Extract Cx and Cy for grid_dict
        df['Cx'] = df['x_mm'] / 1000
        df['Cy'] = df['y_mm'] / 1000
        grid = {
            "Cx": df["Cx"].values.astype("float32"),
            "Cy": df["Cy"].values.astype("float32"),
        }

        return x_t, y_t, grid

    # Train data
    X_train, y_train = [], []
    train_grids = {}
    for case in train_cases:
        x, y, grid = load_case(case)
        X_train.append(x)
        y_train.append(y)
        train_grids[case] = grid

    # Test data
    X_test, y_test = [], []
    test_grids = {}
    for case in test_cases:
        x, y, grid = load_case(case)
        X_test.append(x)
        y_test.append(y)
        test_grids[case] = grid

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "grid_dict": {
            "train": train_grids,
            "test": test_grids
        }
    }

# === Output paths ===
run_name = config["paths"]["name"]
output_dir = Path(config["paths"]["output_dir"]) / run_name
output_dir.mkdir(parents=True, exist_ok=True)
os.environ["KAN_MODEL_DIR"] = str(output_dir / "model")

# === Load a small sample from experimental data
trial = debug_trials[config['trial_name']]
train_cases = trial['train']
test_cases = trial['test']
input_feats = config["features"]["input"]
output_feats = config["features"]["output"]

#load data from config inputs
data_bundle = load_data(exp_dir, train_cases, test_cases, input_feats, output_feats)
grid_dict = data_bundle['grid_dict']
# === Normalize Inputs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_test_normed, X_train_normed, x_scalers = normalize_x_test_train(data_bundle['X_test'],
                                                                  data_bundle['X_train'], device=device)

# === KDE Plot
plot_input_kde(
    x_train_normed=X_train_normed,
    x_test_normed=X_test_normed,
    train_cases=train_cases,
    test_cases=test_cases,
    config=config,
    save_dir=output_dir / "kde"
)

# === Train Model
y_train= data_bundle['y_train']
y_test= data_bundle['y_test']

y_pred, loss_df, best_model, best_epoch, state_dict, optimizer = runSimple_model(
    X_train_normed, y_train, X_test_normed, config, output_dir, device
)

# === Save predictions
pred_dir = output_dir / "predictions"
pred_dir.mkdir(exist_ok=True)
torch.save(y_pred, pred_dir / "y_pred.pt")
torch.save(y_test, pred_dir / "y_test.pt")

# === Evaluate Test Losses
metrics_dir = output_dir / "metrics"
compute_test_losses_and_save(y_preds=y_pred, y_tests=y_test, config=config,
                              case_names=test_cases, save_dir=metrics_dir)

evaluate_cases(
    pred_list=y_pred,
    truth_list=y_test,
    case_names=test_cases,
    config=config,
    output_dir=metrics_dir
)

from Plotting.plot_debug import plot_debug_triplet
plots_dir = os.path.join(output_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)
for yt, yp, name in zip(y_test, y_pred, test_cases):    
    plot_debug_triplet(
        truth=yt,
        pred=yp,
        case_name=name,
        cx=grid_dict['test'][name]['Cx'],
        cy=grid_dict['test'][name]['Cy'],
        save_dir= plots_dir,
        feature_names=output_feats
    )


# === Organize + Plot
results = organize_rst_results(y_pred, y_test, config)
'''
plot_all_tensor_fields(
    results["pred"],
    results["truth"],
    case_names=test_cases,
    save_dir=output_dir / "predictions",
    grid_dict=grid_dict["test"],
    out_features=output_feats
)'''

# === Optionally plot on training set
if config.get("eval_training_cases", False):
    evaluate_training_cases(
        best_model=best_model,
        x_train_list=X_train_normed,
        y_train_list=y_train,
        case_names=train_cases,
        grid_dict=grid_dict["train"],
        config=config,
        output_dir=output_dir / "train_predictions"
    )

# === Plot loss curve
plot_all_losses(loss_df, output_dir / "loss_plots")

# === Plot dynamic weights if used
if config['training']['loss'].get('dynamic_weights', False):
    plot_weights_history(
        loss_df,
        save_dir=output_dir / "weights",
        filename="dynamic_component_weights.png"
    )

print(f"[✓] Debug run complete — results in {output_dir}")
