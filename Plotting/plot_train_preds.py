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
import torch
from pathlib import Path
from Post_Process.organize_results import organize_rst_results
from Plotting.Plot_preds import plot_all_tensor_fields

def evaluate_training_cases(best_model, x_train_list, y_train_list, case_names, grid_dict, config, output_dir):
    """
    Run inference on training set and plot predicted tensors vs ground truth.

    Args:
        best_model (torch.nn.Module): trained model
        x_train_list (list[Tensor]): training inputs per case
        y_train_list (list[Tensor]): ground truth outputs per case
        case_names (list[str]): case identifiers
        grid_dict (dict): { case_name: (cx, cy) }
        config (dict): config dict with 'features'
        output_dir (Path or str): base output directory
    """
    best_model.eval()
    device = next(best_model.parameters()).device

    # Predict
    y_train_preds = []
    with torch.no_grad():
        for x in x_train_list:
            x = x.to(device)
            pred = best_model(x).cpu()
            y_train_preds.append(pred)

    # Organize and plot
    training_results = organize_rst_results(y_train_preds, y_train_list, case_names, config)

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

    print("✅ Training case predictions plotted and saved to:", train_plot_dir)
