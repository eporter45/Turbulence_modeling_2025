"""
author: @eoporter
Fully Connected Network (FCN) model and training utilities.

This script:
- Defines a configurable FCN architecture with customizable layers, width, dropout, and activations
- Trains the model using physics-informed losses, weighted by case
- Supports optional learning rate scheduler integration
- Logs loss histories and saves best and final model checkpoints
- Returns predictions for test cases
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import copy
from Trials import TRIALS, debug_trials
from train_utils.initialize_crit_sched_opt import initialize_crit_sched_optimizer, step_scheduler_if_enabled
from train_utils.make_loss_weights import LossWeights, summate_loss_groups
from train_utils.loss_tracker import LossTracker
from train_utils.calc_data_phys_const_losses import compute_all_losses
from utils.torch_tensor_utils import vec6_to_sym3, symmetrize, sym3_to_vec6, clamp_eigenvalues, make_traceless

# ------------------------------------------------------------------------
# Activation helpers
# ------------------------------------------------------------------------

def get_activation(name, neg_slope=0.01) -> nn.Module:
    name = name.lower()
    if name in ('relu6','relu'):
        return nn.ReLU()
    elif name == 'leakyrelu':
        return nn.LeakyReLU(negative_slope=neg_slope)
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name in ('linear','identity','none'):
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation: {name}")


# ------------------------------------------------------------------------
# FCN MODEL
# ------------------------------------------------------------------------

class FCN(nn.Module):
    def __init__(self, dropout, input_size, output_size,
                 activation='leakyrelu',
                 layers=10, width=10,
                 neg_slope=0.01,
                 enforce_symmetry=False,
                 enforce_traceless=False):

        super(FCN, self).__init__()

        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.dropout = float(dropout)
        self.num_layers = int(layers)
        self.width = int(width)
        self.activation = activation
        self.neg_slope = float(neg_slope)

        self.activation_fn = get_activation(activation, neg_slope)

        layers_list = []

        # Input layer
        layers_list.append(nn.Linear(self.input_size, self.width))
        layers_list.append(self.activation_fn)
        layers_list.append(nn.Dropout(self.dropout))

        # Hidden layers
        for _ in range(self.num_layers - 2):
            layers_list.append(nn.Linear(self.width, self.width))
            layers_list.append(self.activation_fn)
            layers_list.append(nn.Dropout(self.dropout))

        # Output layer
        layers_list.append(nn.Linear(self.width, self.output_size))

        self.net = nn.Sequential(*layers_list)

        # Proper initialization
        for m in self.net:
            if isinstance(m, nn.Linear):
                if self.activation == 'leakyrelu':
                    nn.init.kaiming_normal_(m.weight, a=self.neg_slope, nonlinearity='leaky_relu')
                elif self.activation in {'tanh','sigmoid'}:
                    nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain(self.activation))
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        FCN forward pass with optional symmetry and traceless constraints.
        Output shape remains (..., 6).
        """
        y = self.net(x)  # raw output (...,6)

        # reshape to symmetric 3x3
        T = vec6_to_sym3(y)

        # conditional enforcement
        if getattr(self, "enforce_symmetry", False):
            T = symmetrize(T)

        if getattr(self, "enforce_traceless", False):
            T = make_traceless(T)

        # flatten back to 6-vector
        y_out = sym3_to_vec6(T)
        return y_out


# ------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------

def resolve_train_case_names(config, debug=False):
    trial_val = config['trial_name']
    if isinstance(trial_val, dict):
        return trial_val['train']
    elif debug:
        return debug_trials[trial_val]['train']
    return TRIALS[trial_val]['train']


def save_model_bundle(model, optimizer, directory, tag='final_model'):
    path = os.path.join(directory, tag)
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, 'model_dict.pth'))
    torch.save(optimizer, os.path.join(path, 'optimizer_dict.pth'))


def make_predictions(model, x_test):
    model.eval()
    with torch.no_grad():
        return [model(x) for x in x_test]


# ------------------------------------------------------------------------
# TRAINING LOOP
# ------------------------------------------------------------------------

def train_fcn_model(model, X_train, y_train, config, directory, device,
                    y_frob_max=None, y_k_max=None):

    criterion, optimizer, scheduler = initialize_crit_sched_optimizer(model, config)

    best_epoch, best_loss = 0, float('inf')
    best_model = copy.deepcopy(model)
    best_state_dict, best_optimizer = None, None

    train_case_names = resolve_train_case_names(config, config['debug'])
    weights = LossWeights(config, train_case_names, y_train, verbose=True)

    if config['training']['loss']['weights']['enabled']:
        print("Initial Loss weights:\n", weights.print_weights())

    loss_tracker = LossTracker(train_case_names, config)
    model.to(device)

    # ------------------- EPOCH LOOP -------------------
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_store = loss_tracker._init_epoch_storage()

        # Loop over training cases
        for idx, (x, y_true) in enumerate(zip(X_train, y_train)):
            name = train_case_names[idx]

            dataset = torch.utils.data.TensorDataset(x.to(device), y_true.to(device))
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config['training']['batch_size'],
                shuffle=True
            )

            n_points = 0

            for x_b, y_b in loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                y_pred = model(x_b)

                # Compute physics/data losses
                loss_dict = compute_all_losses(
                    config, y_pred, y_b, criterion,
                    epoch=epoch, y_frob_max=y_frob_max, y_k_max=y_k_max
                )
                weighted_loss = weights.apply(loss_dict, name)
                loss = summate_loss_groups(weighted_loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bs = x_b.size(0)
                n_points += bs
                weighted_loss['net'] = loss.detach()
                loss_tracker.update_batch(epoch_store, name, weighted_loss, bs)

            loss_tracker.finalize_case(epoch_store, name, n_points=n_points)

        # End epoch
        loss_tracker.finalize_epoch(epoch_store)

        net_loss = epoch_store['total']['net']
        net_data_loss = epoch_store['total']['data']['net']
        net_phys_loss = epoch_store['total']['phys']['net']
        net_const_loss = epoch_store['total']['constraint']['net']

        # Track best
        if net_loss < best_loss:
            best_loss = net_loss
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            best_state_dict = copy.deepcopy(model.state_dict())
            best_optimizer = copy.deepcopy(optimizer.state_dict())

        # Logs
        if epoch % config['training']['eval_every'] == 0:
            print(
                f"[Epoch {epoch}/{config['training']['epochs']}] "
                f"Net: {net_loss:.3e} | Data: {net_data_loss:.3e} | "
                f"Phys: {net_phys_loss:.3e} | Const: {net_const_loss:.3e}"
            )

        step_scheduler_if_enabled(scheduler, config, epoch, net_loss)

        if epoch % 250 == 0 and epoch != 0:
            save_model_bundle(model, optimizer, directory, tag=f"epoch_{epoch}")

    # Save final
    save_model_bundle(best_model, best_optimizer, directory, tag='best_model')
    save_model_bundle(model, optimizer, directory, tag='final_model')

    loss_df = loss_tracker.get_all_histories()

    return model, loss_df, best_model, best_epoch, best_state_dict, optimizer


# ------------------------------------------------------------------------
# RUN FCN MODEL (MAIN ENTRY POINT)
# ------------------------------------------------------------------------

def runSimple_model(data_bundle, config, directory, device, y_frob_max, y_k_max):

    X_train = data_bundle['x_train']
    y_train = data_bundle['y_train']
    X_test = data_bundle['x_test']

    model = FCN(
        dropout=config['model']['dropout'],
        input_size=X_train[0].shape[1],
        output_size=y_train[0].shape[1],
        activation=config['model']['activation'],
        layers=config['model']['layers'],
        width=config['model']['width'],
        enforce_symmetry=config['model'].get('enforce_symmetry', False),
        enforce_traceless=config['model'].get('enforce_traceless', False),
    ).to(device)

    print(f"[MODEL] Layers: {model.num_layers}")
    print(f"[MODEL] Width: {model.width}")
    print(f"[MODEL] Parameters: {sum(p.numel() for p in model.parameters())}")

    # Train
    final_model, loss_df, best_model, best_epoch, best_state_dict, optimizer = train_fcn_model(
        model, X_train, y_train, config, directory, device
    )

    # Predictions
    predictions = make_predictions(best_model.to(device), X_test)
    fin_preds = make_predictions(final_model.to(device), X_test)

    # FINAL RETURN — **NO PARENTHESES**
    return fin_preds, predictions, loss_df, best_model, best_epoch, best_state_dict, optimizer
