# -*- coding: utf-8 -*-
"""
PyTorch TBNN implementation — UPDATED Feb 2025 (clean + debug safe)

Key improvements:
 - Auto-detects invariant and basis dims
 - Optional y normalization (based on config)
 - Debug-safe with assert checks and shape validation
 - Clean training loop
 - Symmetry, tracelessness, and optional eigenvalue clamping
"""

import numpy as np
import torch
import torch.nn as nn
import os
import copy

from train_utils.initialize_crit_sched_opt import (
    initialize_crit_sched_optimizer,
    step_scheduler_if_enabled
)
from train_utils.make_loss_weights import LossWeights, summate_loss_groups
from train_utils.loss_tracker import LossTracker
from train_utils.calc_data_phys_const_losses import compute_all_losses

# tensor utilities
from utils.torch_tensor_utils import (
    symmetrize, make_traceless,
    sym3_to_vec6, clamp_eigenvalues
)

# -------------------------------------------------------------------------
#                           TBNN CORE (MLP)
# -------------------------------------------------------------------------
class TBNNCore(nn.Module):
    """
    Pure feedforward part of TBNN:
    invariants → coefficients g_n
    """

    def __init__(self, n_scalars, n_tensors, hidden_layers,
                 activation=nn.ReLU, dropout=0.0):
        super().__init__()

        layers = []
        in_dim = n_scalars

        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h

        layers.append(nn.Linear(in_dim, n_tensors))  # final g_n coefficients
        self.net = nn.Sequential(*layers)

    def forward(self, invariants):
        return self.net(invariants)   # [N, n_tensors]


# -------------------------------------------------------------------------
#                           TBNN MODEL WRAPPER
# -------------------------------------------------------------------------
class TBNNModel(nn.Module):
    """
    TBNN:
      - invariants → g
      - a_pred = g_n * T^(n)
      - symmetry + traceless enforcement
      - return vec6 form (xx,xy,xz,yy,yz,zz)
    """

    def __init__(self,
                 input_dim_invariants,
                 tensor_basis_dim,
                 hidden_dims,
                 activation=nn.ReLU,
                 dropout=0.0,
                 enforce_symmetry=True,
                 enforce_traceless=True,
                 clamp_min_eig=None):

        super().__init__()

        self.enforce_symmetry = enforce_symmetry
        self.enforce_traceless = enforce_traceless
        self.clamp_min_eig = clamp_min_eig

        self.core = TBNNCore(
            n_scalars=input_dim_invariants,
            n_tensors=tensor_basis_dim,
            hidden_layers=hidden_dims,
            activation=activation,
            dropout=dropout
        )

    def forward(self, invariants, tensor_basis):
        """
        invariants    : (N, n_scalars)
        tensor_basis  : (N, n_tensors, 9)
        returns       : (N, 6) vec6 anisotropy tensor
        """

        # Debugging checks
        assert invariants.dim() == 2, "Invariants must be (N, d_inv)"
        assert tensor_basis.dim() == 3, "Tensor basis must be (N, d_basis, 9)"
        assert tensor_basis.shape[2] == 9, "Tensor basis last dim must be 9"

        g = self.core(invariants)              # (N, n_tensors)

        # contraction over basis tensors
        y = torch.einsum("nb,nbc->nc", g, tensor_basis)  # (N, 9)
        N = y.shape[0]
        y = y.view(N, 3, 3)

        if self.enforce_symmetry:
            y = symmetrize(y)

        if self.clamp_min_eig is not None:
            y = clamp_eigenvalues(y, min_eig=self.clamp_min_eig)

        if self.enforce_traceless:
            y = make_traceless(y)

        return sym3_to_vec6(y)
    def forward_with_coeffs(self, invariants, tensor_basis):
        """
        Returns BOTH:
          - g:    (N, n_tensors)  learned coefficients
          - a6:   (N, 6)          anisotropy vector prediction
        Used ONLY for evaluation/visualization.
        """

        assert invariants.dim() == 2
        assert tensor_basis.dim() == 3
        assert tensor_basis.shape[2] == 9

        # compute coefficients
        g = self.core(invariants)          # (N, n_tensors)

        # contract into anisotropy
        y = torch.einsum("nb,nbc->nc", g, tensor_basis)
        N = y.shape[0]
        y = y.view(N, 3, 3)

        if self.enforce_symmetry:
            y = symmetrize(y)
        if self.clamp_min_eig is not None:
            y = clamp_eigenvalues(y, min_eig=self.clamp_min_eig)
        if self.enforce_traceless:
            y = make_traceless(y)

        a6 = sym3_to_vec6(y)
        return g, a6


# -------------------------------------------------------------------------
#                           SAVE HELPERS
# -------------------------------------------------------------------------
def save_model_bundle(state_dict, optimizer_state, directory, tag='final'):
    out = os.path.join(directory, tag)
    os.makedirs(out, exist_ok=True)
    torch.save(state_dict, os.path.join(out, "model_state_dict.pth"))

def symmetrize_flatten(a_vec6):
    """
    Convert vec6 → symmetrize → vec6
    Works case-wise on (N, 6) tensors.
    """
    from utils.torch_tensor_utils import vec6_to_sym3, symmetrize, sym3_to_vec6

    A = vec6_to_sym3(a_vec6)     # (N,3,3)
    A = symmetrize(A)            # enforce symmetry
    return sym3_to_vec6(A)       # return (N,6)

# -------------------------------------------------------------------------
#                            TRAINING LOOP
# -------------------------------------------------------------------------
def train_tbnn(model, X_inv, X_tb, Y_true,
               config, directory, device,
               y_frob_max=1.0, y_k_max=1.0):

    criterion, optimizer, scheduler = initialize_crit_sched_optimizer(model, config)
    train_case_names = [f"case{i}" for i in range(len(X_inv))]
    weights = LossWeights(config, train_case_names, Y_true, verbose=True)
    tracker = LossTracker(train_case_names, config)

    best_loss = float('inf')
    best_model = None
    best_state = None
    best_epoch = 0

    for epoch in range(config['training']['epochs']):
        model.train()
        ep_store = tracker._init_epoch_storage()

        for idx, (inv, tb, yt) in enumerate(zip(X_inv, X_tb, Y_true)):
            inv = inv.to(device)
            tb = tb.to(device)
            yt = yt.to(device)

            y_pred = model(inv, tb)

            loss_dict = compute_all_losses(
                config, y_pred, yt,
                criterion, epoch=epoch,
                y_frob_max=y_frob_max,
                y_k_max=y_k_max
            )
            #print("DEBUG loss_dict:", loss_dict)
            #print("DEBUG loss dict data crit:", loss_dict['data_crit'].shape)
            wloss = weights.apply(loss_dict, train_case_names[idx])
            loss = summate_loss_groups(wloss)
            print("[DEBUG] loss dict :", loss_dict)
            print("[DEBUG] loss dict shape:", loss_dict['data_crit'].shape)
            # FORCE wrap floats as tensors (gradient-safe zero-loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tracker.update_batch(ep_store, train_case_names[idx], wloss, inv.shape[0])

        tracker.finalize_epoch(ep_store)

        net_loss = ep_store["total"]["net"]
        if net_loss < best_loss:
            best_loss = net_loss
            best_epoch = epoch
            best_model = copy.copy(model)
            best_state = copy.deepcopy(model.state_dict())

        if epoch % config['training']['eval_every'] == 0:
            print(f"[epoch {epoch}] net loss = {net_loss:.3e}")

        step_scheduler_if_enabled(scheduler, config, epoch, net_loss)

    save_model_bundle(best_state, optimizer.state_dict(), directory, tag="best_model")
    save_model_bundle(model.state_dict(), optimizer.state_dict(), directory, tag="final_model")

    return model, tracker.get_all_histories(), best_model, best_epoch, best_state, optimizer


# -------------------------------------------------------------------------
#                                RUN TBNN
# -------------------------------------------------------------------------
def run_tbnn_model(
    config, data_dict, directory, device,
    y_frob_max=None, y_k_max=None
):

    print("\n🚀 Running TBNN training...\n")

    # Default normalization scalars
    y_frob_max = y_frob_max if (config['features'].get('y_norm', False)) else 1.0
    y_k_max    = y_k_max    if (config['features'].get('y_norm', False)) else 1.0

    # ---------------------------------------------------
    # Load TRAIN tensors
    # ---------------------------------------------------
    X_train_inv, X_train_tb, Y_train = [], [], []
    for case, case_data in data_dict['train'].items():
        inv = case_data['invariants']
        tb  = case_data['tensor_basis']
        y   = case_data['out_true']

        if config['features'].get('y_norm', False):
            y = y / y_frob_max

        X_train_inv.append(torch.tensor(inv, dtype=torch.float32))
        X_train_tb.append(torch.tensor(tb, dtype=torch.float32))
        Y_train.append(torch.tensor(y, dtype=torch.float32))

    # ---------------------------------------------------
    # Load TEST tensors
    # ---------------------------------------------------
    X_test_inv, X_test_tb, Y_test = [], [], []
    for case, case_data in data_dict['test'].items():
        inv = case_data['invariants']
        tb  = case_data['tensor_basis']
        y   = case_data['out_true']

        if config['features'].get('y_norm', False):
            y = y / y_frob_max

        X_test_inv.append(torch.tensor(inv, dtype=torch.float32))
        X_test_tb.append(torch.tensor(tb, dtype=torch.float32))
        Y_test.append(torch.tensor(y, dtype=torch.float32))

    # ------------------------------------------
    # Auto-detect dims
    # ------------------------------------------
    example_inv = X_train_inv[0]
    example_tb  = X_train_tb[0]

    input_dim_invariants = example_inv.shape[1]
    tensor_basis_dim     = example_tb.shape[1]

    print(f"[INFO] Auto-detected invariant dim     : {input_dim_invariants}")
    print(f"[INFO] Auto-detected tensor basis dim  : {tensor_basis_dim}")

    # Safety checks
    assert input_dim_invariants > 0
    assert tensor_basis_dim > 0
    assert example_tb.shape[2] == 9, "Tensor basis must be (N, d_basis, 9)"

    # ------------------------------------------
    # Move tensors to device
    # ------------------------------------------
    X_train_inv = [t.to(device) for t in X_train_inv]
    X_train_tb  = [t.to(device) for t in X_train_tb]
    Y_train     = [t.to(device) for t in Y_train]

    X_test_inv = [t.to(device) for t in X_test_inv]
    X_test_tb  = [t.to(device) for t in X_test_tb]
    Y_test     = [t.to(device) for t in Y_test]

    # ------------------------------------------
    # Instantiate model
    # ------------------------------------------
    model = TBNNModel(
        input_dim_invariants=input_dim_invariants,
        tensor_basis_dim=tensor_basis_dim,
        hidden_dims=config['model']['hidden_dims'],
        activation=nn.ReLU,
        enforce_symmetry=True,
        enforce_traceless=True,
        clamp_min_eig=None
    ).to(device)

    print("[INFO] Example shapes:")
    print(" invariants:", example_inv.shape)
    print(" tensor basis:", example_tb.shape)
    print(" out_true:", Y_train[0].shape)

    # ---------------------------------------------------
    # Train
    # ---------------------------------------------------
    model, loss_df, best_model, best_epoch, best_state, optimizer = train_tbnn(
        model,
        X_train_inv, X_train_tb, Y_train,
        config, directory, device,
        y_frob_max=y_frob_max,
        y_k_max=y_k_max
    )

    # ---------------------------------------------------
    # Evaluate
    # ---------------------------------------------------
    best_model.eval()
    out_preds = []
    coeffs = []
    with torch.no_grad():
        for inv, tb in zip(X_test_inv, X_test_tb):
            g, a6 = best_model.forward_with_coeffs(inv, tb)
            out_preds.append(a6.cpu())
            coeffs.append(g.cpu())
    preds = {'tensor': out_preds, 'coeffs': coeffs}
    return preds, model, loss_df, best_model, best_epoch, best_state, optimizer
