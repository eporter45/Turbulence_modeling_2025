# -*- coding: utf-8 -*-
"""
Module for computing test losses on the model predictions

Created on Wed Jul 16 12:03:06 2025

@author: eoporter
"""

from pathlib import Path
from train_utils.calc_data_phys_const_losses import compute_all_losses
import json
import torch

def tensor_to_num(obj):
    if isinstance(obj, dict):
        return {k: tensor_to_num(v) for k, v in obj.items()}
    elif torch.is_tensor(obj):
        return obj.item()  # convert single-element tensor to float
    elif isinstance(obj, (list, tuple)):
        return [tensor_to_num(i) for i in obj]
    else:
        return obj

# then, inside your loop:



def compute_test_losses_and_save(y_preds, y_tests, config, case_names, save_dir, y_frob_max, y_k_max,  fin_best=''):
    """
    Computes and saves test loss breakdown (data, phys, constraint) for all test cases.
    """
    from train_utils.initialize_crit_sched_opt import initialize_criterion
    criterion = initialize_criterion(config)
    all_loss_dicts = []
    total_data_loss = 0.0
    total_phys_loss = 0.0
    total_constraint_loss = 0.0
    num_cases = len(y_preds)

    for pred, truth, name in zip(y_preds, y_tests, case_names):
        loss_dict = compute_all_losses(config, pred, truth, criterion, y_frob_max=y_frob_max, y_k_max= y_k_max)

        data_loss = 0.0
        phys_loss = 0.0
        constraint_loss = 0.0

        if loss_dict.get('data'):
            data_loss = sum(loss_dict['data'].values())

        if loss_dict.get('phys'):
            phys_loss = sum(loss_dict['phys'].values())

        if loss_dict.get('constraint'):
            constraint_loss = sum(loss_dict['constraint'].values())

        # Safely convert to float
        data_loss = data_loss.item() if torch.is_tensor(data_loss) else float(data_loss)
        phys_loss = phys_loss.item() if torch.is_tensor(phys_loss) else float(phys_loss)
        constraint_loss = constraint_loss.item() if torch.is_tensor(constraint_loss) else float(constraint_loss)

        total_data_loss += data_loss
        total_phys_loss += phys_loss
        total_constraint_loss += constraint_loss

        all_loss_dicts.append({
            'case': name,
            'data_loss': data_loss,
            'phys_loss': phys_loss,
            'constraint_loss': constraint_loss,
            'losses': tensor_to_num(loss_dict)  # recursively convert tensors
        })

    summary = {
        'average_data_loss': total_data_loss / num_cases,
        'average_phys_loss': total_phys_loss / num_cases,
        'average_constraint_loss': total_constraint_loss / num_cases,
        'per_case_losses': all_loss_dicts
    }

    save_path = Path(save_dir) / f"{fin_best}_test_loss_info.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Test loss breakdown saved to: {save_path}")



