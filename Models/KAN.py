'''
author @eoporter
KAN Training Pipeline

This script defines the training pipeline for Kernel-based Artificial Neurons (KANs),
specifically using the `KAN_me` class from `KAN_lib.kan.Multikan`. It includes:

1. Model Initialization:
   - Builds a customizable KAN model architecture based on `shape`, `grid range`, and spline order.

2. Training Utilities:
   - Supports both standard training and training with learning rate schedulers.
   - Handles per-case loss computation, dynamic weighting, and logging.
   - Tracks best-performing model based on validation losses.
'''

import os
import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from KAN_lib.kan.MultKAN import KAN_me as KAN
from Trials import TRIALS
from train_utils.initialize_crit_sched_opt import initialize_crit_sched_optimizer, step_scheduler_if_enabled
from train_utils.make_loss_weights import LossWeights, summate_loss_groups
from train_utils.loss_tracker import LossTracker
from train_utils.calc_data_phys_const_losses import compute_all_losses
from utils.torch_tensor_utils import (
    sym3_to_vec6,
    symmetrize,
    vec6_to_sym3,
    clamp_eigenvalues,
    make_traceless,
)

# ---------------------------------------------------------------------
#                       KAN MODEL WRAPPER
# ---------------------------------------------------------------------

class KANModel(nn.Module):
    """
    Wraps the raw KAN network and enforces:
    - optional symmetry
    - optional tracelessness
    - flattening to vec6
    """
    def __init__(self, kan_core,
                 enforce_symmetry=True,
                 enforce_traceless=True,
                 clamp_min_eig=None):   # NEW: optional SPD eigenvalue clamping
        super().__init__()
        self.kan = kan_core
        self.enforce_symmetry = enforce_symmetry
        self.enforce_traceless = enforce_traceless
        self.clamp_min_eig = clamp_min_eig

    def forward(self, x):
        # raw KAN output → [N,9]
        Y = self.kan(x)
        N = Y.shape[0]

        # reshape to 3×3
        Y = Y.view(N, 3, 3)

        # ---- Symmetry ----
        if self.enforce_symmetry:
            Y = symmetrize(Y)

        # ---- Eigenvalue clamping (SPD projection step) ----
        if self.clamp_min_eig is not None:
            Y = clamp_eigenvalues(Y, min_eig=self.clamp_min_eig)

        # ---- Zero-trace ----
        if self.enforce_traceless:
            Y = make_traceless(Y)

        # ---- Flatten to vec6 ----
        return sym3_to_vec6(Y)


# ---------------------------------------------------------------------
#                       MODEL INITIALIZATION
# ---------------------------------------------------------------------

def initialize_kan(n_in, n_out, config, device):

    def sanitize_shape(shape):
        return [s[0] if isinstance(s, list) and len(s) == 2 and s[1] else s
                for s in shape]

    print("[DEBUG][Initialize_kan] Raw model shape:", config['model']['shape'], flush=True)
    shape = sanitize_shape(config['model']['shape'])
    shape = [n_in] + shape + [n_out]   # KAN width definition

    print('[INFO] Sanitized KAN Shape:', shape, flush=True)

    # --- raw KAN core ---
    kan_core = KAN(
        width=shape,
        grid=config['model'].get('spline_order', 3),
        seed=config['model'].get('seed', 42),
        grid_range=config['model'].get('grid_range', [-1.2, 1.2])
    )

    # --- wrap with tensor constraints ---
    model = KANModel(
        kan_core,
        enforce_symmetry=config['model'].get('enforce_symmetry', True),
        enforce_traceless=config['model'].get('enforce_traceless', True),
    )

    print('[INFO] KAN model initialized (wrapped)', flush=True)
    return model


# ---------------------------------------------------------------------
#                          HELPER FUNCTIONS
# ---------------------------------------------------------------------

def save_model_bundle(state_dict, optimizer, directory, tag='final model'):
    path = os.path.join(directory, tag)
    os.makedirs(path, exist_ok=True)
    torch.save(state_dict, os.path.join(path, 'model_dict.pth'))
    # torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer_dict.pth'))


def resolve_train_case_names(config):
    trial_val = config['trial_name']
    if isinstance(trial_val, dict):
        return trial_val['train']
    return TRIALS[trial_val]['train']


# ---------------------------------------------------------------------
#                           TRAINING LOOP
# ---------------------------------------------------------------------

def train_kan_model(model, X_train, y_train, config, directory,
                    device, y_frob_max=None, y_k_max=None):

    criterion, optimizer, scheduler = initialize_crit_sched_optimizer(model, config)

    best_epoch, best_loss = 0, float('inf')
    best_model = copy.deepcopy(model)
    best_state_dict, best_optimizer = None, None

    train_case_names = resolve_train_case_names(config)
    weights = LossWeights(config, train_case_names, y_train, verbose=True)

    if config['training']['loss']['weights']['enabled']:
        print("Initial Loss weights:\n", weights.print_weights())

    loss_tracker = LossTracker(train_case_names, config)
    model.to(device)

    # ------------------ Epoch Loop ------------------
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_storage = loss_tracker._init_epoch_storage()

        # ---------- Loop over cases ----------
        for idx, (x, y_true) in enumerate(zip(X_train, y_train)):
            name = train_case_names[idx]

            dataset = torch.utils.data.TensorDataset(
                x.to(device), y_true.to(device)
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config['training']['batch_size'],
                shuffle=True
            )

            n_points = 0

            # ---------- Mini-batch loop ----------
            for x_b, y_b in loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                y_pred = model(x_b)

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
                loss_tracker.update_batch(epoch_storage, name, weighted_loss, bs)

            loss_tracker.finalize_case(epoch_storage, name, n_points=n_points)

        # ---------- End of epoch ----------
        loss_tracker.finalize_epoch(epoch_storage)
        net_loss = epoch_storage['total']['net']
        net_data_loss = epoch_storage['total']['data']['net']
        net_phys_loss = epoch_storage['total']['phys']['net']
        net_const_loss = epoch_storage['total']['constraint']['net']

        if net_loss < best_loss:
            best_loss = net_loss
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            best_state_dict = copy.deepcopy(model.state_dict())
            best_optimizer = copy.deepcopy(optimizer.state_dict())

        if epoch % config['training']['eval_every'] == 0:
            print(
                f"[Epoch {epoch}/{config['training']['epochs']}]   "
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


# ---------------------------------------------------------------------
#                         RUN KAN MODEL
# ---------------------------------------------------------------------

def runKAN_model(data_bundle, config, directory, device,
                 y_frob_max=None, y_k_max=None):

    # Extract data
    x_train = data_bundle['x_train']
    x_test = data_bundle['x_test']
    y_train = data_bundle['y_train']

    # Reproducibility
    seed = config['training'].get('seed', 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    input_size = x_train[0].shape[1]
    output_size = y_train[0].shape[1]

    # Build model
    model = initialize_kan(
        n_in=input_size,
        n_out=output_size,
        config=config,
        device=device
    )

    print(f"[MODEL] Shape: {config['model']['shape']}")
    print(f"[MODEL] Grid range: {config['model'].get('grid_range', [-1.2, 1.2])}")
    print(f"[MODEL] Spline order: {config['model'].get('spline_order', 3)}")
    print(f"[MODEL] Total parameters: {sum(p.numel() for p in model.parameters())}")

    # --------- Train ----------
    fin_model, loss_df, best_model, best_epoch, best_state_dict, optimizer = train_kan_model(
        model, x_train, y_train, config, directory, device,
        y_frob_max=y_frob_max, y_k_max=y_k_max
    )

    # --------- Predictions ----------
    best_model.to(device)
    best_model.eval()
    with torch.no_grad():
        predictions = [best_model(x.to(device)).cpu() for x in x_test]

    fin_model.to(device)
    fin_model.eval()
    with torch.no_grad():
        final_predictions = [fin_model(x.to(device)).cpu() for x in x_test]

    # ***************
    # FINAL RETURN — no parentheses
    # ***************
    return final_predictions, predictions, loss_df, best_model, best_epoch, best_state_dict, optimizer
