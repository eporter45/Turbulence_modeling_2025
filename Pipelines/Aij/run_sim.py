# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 19:24:32 2025

@author: eoporter
"""
import os
import sys
import random
# Dynamically resolve project root (Phys_KAN root dir)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Make sure Phys_KAN root is in sys.path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Pipelines.Aij.Simulation import main
from PreProcess.resolve_root import resolve_config_paths  

# Show paths
print(f"[INFO] cwd         : {os.getcwd()}")
print(f"[INFO] PROJECT_ROOT: {PROJECT_ROOT}")

#data_dir = os.path.join(PROJECT_ROOT, 'Data', 'Shear_mixing')
print(f'cwd: {os.getcwd()}')
print(f'Proj_root: {PROJECT_ROOT}')
cfg = {
       
    "debug": False,                         # Optional debug mode to skip full evaluation in evaluate_cases
    "trial_name": "single_inter_c2_f2",        # Replace with a key from TRIALS
    "eval_training_cases": True,             # Whether to evaluate predictions on training cases
    "seed": 42,
    
    "paths": {
        "name": "new_losses_aij_t1",              # name the run/output dir
        "output_dir": "Pipelines/Aij/outputs",          # Base output directory
        "data_dir": ''   # Base data directory
    },

    "features": {
        "dnd": "nondim",                    # dimensional or nondim data ['dim', 'nondim']
        "grad_type": "MLS",                 # RANS feat eng grad type: [MLS, og]
        "input": 'FS9',
        "norm": 'global',
        #"FS1",    # e.g., FS1, FS2 read Features.make_featursets.py
        "output": ['a_xx', 'a_xy', 'a_yy', 'a_xz', 'a_yz', 'a_zz', 'tke'],  # Model output targets
        "y_norm": False,               # Norm method read Preprocess.load_and_norm.py
        'denorm_loss': False,
        "in_is_out": False,                 # for Debugging, if we want to recreate data
        'trim_z': True,
        "save_kde": True                    # Whether to plot/save KDE plots
    },

    "model": {
        "type": "fcn",                      # Options: "kan", "fcn"
        #FCN specific Params
        "dropout": 0.005,                    # Dropout probability (only for FCN)
        "activation": "leakyrelu",          # Activation function: relu, leakyrelu, tanh, etc.
        "layers": 15,                       # Number of layers (for FCN)
        "width": 10,                     # Width of each hidden layer (for FCN)

        # KAN-specific parameters:
        "shape": [[5,5], [5,5], [5,5]],      # Example layer widths for KAN initialization
        "spline_order": 3,                # Spline order for KAN grid
        "seed": 42,                      # Random seed for reproducibility in KAN
        "grid_range": [-0.2, 1.2]        # Range for KAN grid
    },

    "training": {
        "criterion": "mse",  # Options: "mse", "mae", "rel_mse"
        #for smooth L1 loss
        "beta": 1.0,
        
        "optimizer": "adam",      # Optimizer type: "adam", "adamw", "rmsprop"
        "lr": 0.01,              # Learning rate
        "lambda": 0.00005,         # Weight decay (can be "" if unused)
        "alpha": 0.99,            # Alpha for RMSprop (ignored for Adam)
    
        "epochs": 500,            # Total number of epochs
        "batch_size": 2048,        # Samples per batch
        "eval_every": 20,         # Evaluation/logging frequency
        "seed": 42,               # Random seed
    
        "scheduler": {
            "enabled": True,
            "delay": 500,
            "type": "step",  # Options: reduce_on_plateau, step, multistep
            #for step
            'step_size': 300,
            'gamma': 0.5,
            # for reduce
            "patience": 300,
            "factor": 0.65,
            "min_lr": 1e-8,
            "cooldown": 100,
            # for annealing
            'T_0': 200,
            'T_mult': 2,
            'eta_min': 1e-9,
        },
        "loss":{
            #"type": "mse",  # Loss function: "mse", "mae", etc.
            "warmup":{ 'eigen': 1000,
                      'invariants': 1000
                      },
            "terms": {
                "phys": {
                    "enabled": False,
                    "types": ["log_euclidean"],  # or your actual phys loss type
                        },
                "constraint": {
                    "enabled": True,
                    "types": ['inv_a_comp'],  # or your constraint loss type
                        },
                "data": {
                    "enabled": True,  # Just enable/disable, no type for data
                     "types": ['comp'],
                     "tke_scalar": 5.0,
                        }
                    },       
            "weights": {
                "enabled": True,  # Top-level control over whether to use custom weighting        
                "group": {
                    "enabled": True,
                    "dynamic": False,
                    "data": 01.0,
                    "phys": 0.050,
                    "constraint": 0.00010
                        },      
                "case": {
                    "enabled": False,
                    "dynamic": False
                        },        
                "data": {
                    "enabled": True,
                    'tke_scalar': 5.0,
                    "dynamic": False
                        },        
                "phys": {
                    "enabled": False,
                    "dynamic": False
                        },       
                "constraint": {
                    "enabled": False,
                    "dynamic": False
                        },        
                "phase_switch": {
                    "enabled": False,
                    "loss_key": "",
                    "threshold_percent": 0.1,
                    "pre_switch_weight": 0.01,
                    "post_switch_weight": 1.0
                                }
                    }
        },

    },

}
#peter destrow
#moba xterm mobile 24.3

cfg = resolve_config_paths(cfg, PROJECT_ROOT)
#resolve random seeds
import numpy as np
import torch 

seed = cfg['seed']
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False





main(cfg)
