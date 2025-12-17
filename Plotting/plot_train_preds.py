# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 11:59:17 2025

@author: eoporter
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 11:56:57 2025

@author: eoporter
"""
from pathlib import Path
import torch
import os
from Plotting.Plot_preds import plot_all_tensor_fields
from Post_Process.organize_results import organize_rst_results


def evaluate_training_cases(best_model, x_train_list, y_train_list, case_names,
                            grid_dict, config, output_dir, model_type=None, device='cpu'):
    """
    Evaluates model predictions on training data for visual validation.
    Works for FCN, KAN, and TBNN models.

    Args
    ----
        best_model (torch.nn.Module): trained model
        x_train_list (list[Tensor] or list[tuple]): training inputs per case
        y_train_list (list[Tensor]): ground truth outputs per case
        case_names (list[str]): case identifiers
        grid_dict (dict): { case_name: (Cx, Cy) }
        config (dict): config dict with 'features'
        output_dir (Path or str): base output directory
        model_type (str, optional): one of ['fcn', 'kan', 'tbnn']
        device (str, optional): computation device
    """
    # --- Setup ---
    device = device or next(best_model.parameters()).device
    best_model.to(device)
    best_model.eval()

    y_train_preds = []
    model_type = model_type or config['model']['type'].lower()

    # --- Forward predictions ---
    with torch.no_grad():
        for x in x_train_list:
            if model_type in ['fcn', 'kan']:
                # Single-input models
                x = x.to(device)
                pred = best_model(x).cpu()

            elif model_type == 'tbnn':
                # Tensor-basis model expects (invariants, tensor_basis)
                if isinstance(x, tuple) and len(x) == 2:
                    inv, tb = x
                    inv, tb = inv.to(device), tb.to(device)
                    pred = best_model(inv, tb).cpu()
                else:
                    raise TypeError(
                        f"[ERROR] Expected (invariants, tensor_basis) tuple for TBNN, got {type(x)}"
                    )

            else:
                raise ValueError(f"[ERROR] Unsupported model type: {model_type}")

            y_train_preds.append(pred)

    # --- Organize and plot results ---
    print("[INFO] Organizing training results...")
    training_results = organize_rst_results(
        y_train_preds, y_train_list, case_names, config
    )

    train_plot_dir = Path(output_dir) / "training_preds"
    train_plot_dir.mkdir(parents=True, exist_ok=True)

    plot_all_tensor_fields(
        pred_dict=training_results['pred'],
        truth_dict=training_results['truth'],
        case_names=case_names,
        grid_dict=grid_dict,
        save_dir=train_plot_dir,
        config=config
    )

    print(f"✅ Training case predictions plotted and saved to: {train_plot_dir}")

    # --- Optional: compute quick metrics per case ---
    import torch.nn.functional as F
    from Models.TBNN import symmetrize_flatten  # ensure TBNN outputs are symmetrized

    metrics = {}
    for name, y_true, y_pred in zip(case_names, y_train_list, y_train_preds):
        # --- Fix shape mismatch for TBNN ---
        if model_type == 'tbnn' and y_pred.shape[1] == 9:
            y_pred = symmetrize_flatten(y_pred)

        # Compute RMSE and R²
        rmse = torch.sqrt(F.mse_loss(y_pred, y_true.cpu())).item()
        ss_res = torch.sum((y_true.cpu() - y_pred) ** 2)
        ss_tot = torch.sum((y_true.cpu() - torch.mean(y_true.cpu())) ** 2)
        r2 = 1 - ss_res / ss_tot

        metrics[name] = {'RMSE': rmse, 'R2': r2.item()}
        print(f"[TRAIN] {name}: RMSE={rmse:.3e}, R²={r2:.3f}")

    # --- Return both results and metrics ---
    return training_results, metrics

