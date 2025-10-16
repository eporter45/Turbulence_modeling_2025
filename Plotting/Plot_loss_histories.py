# -*- coding: utf-8 -*-
"""
Plot Training Losses and Their Components Over Epochs

Functions to visualize training loss histories including:
- Total network loss over epochs
- Per-case total losses
- Grouped losses (data, physics, constraint)
- Detailed component losses within each group

Saves plots to specified directory.

Created on Wed Jul 16 13:42:20 2025

@author: eoporter
"""

import os
import matplotlib.pyplot as plt
import torch


def plot_single_plot(history_dict, save_path, title=None, ylabel="Loss", xlabel="Epoch", log_scale=True):
    """
    Plot a dictionary of {label: list_of_values}.
    The values are assumed to be ordered per epoch.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for label, values in history_dict.items():
        clean_values = [v.detach().cpu().item() if torch.is_tensor(v) else v for v in values]
        epochs = list(range(len(clean_values)))
        ax.plot(epochs, clean_values, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if log_scale:
        ax.set_yscale('log')
        
    if title:
        ax.set_title(title)
        
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)

    
    
def plot_all_losses(histories, save_dir):
    """
    Plot:
        1. Total net loss
        2. Per-case total net losses
        3. Total group losses (data, phys, constraint)
        4. Component losses within each group
    """
    # 1. Total net loss
    total_net = histories["total"]["net"]
    plot_single_plot(
        {"Total Loss": total_net},
        save_path=os.path.join(save_dir, "total_loss.png"),
        title="Total Training Loss"
    )
    
    # 2. Per-case total net losses
    case_histories = {case: hist["net"] for case, hist in histories.items() if case != "total"}
    if case_histories:
        plot_single_plot(
            case_histories,
            save_path=os.path.join(save_dir, "case_total_losses.png"),
            title="Total Loss Per Case"
        )
    
    # 3. Total group losses (data, phys, constraint)
    group_loss_dict = {
        group.capitalize(): histories["total"][group]["net"]
        for group in ["data", "phys", "constraint"]
        if "net" in histories["total"][group]
    }
    if group_loss_dict:
        plot_single_plot(
            group_loss_dict,
            save_path=os.path.join(save_dir, "group_losses.png"),
            title="Group Losses"
        )
    
    # 4. Component losses within each group
    for group in ["data", "phys", "constraint"]:
        group_dict = histories["total"].get(group, {})
        component_dict = {
            comp: losses for comp, losses in group_dict.items()
            if comp != "net"
        }
        if component_dict:
            plot_single_plot(
                component_dict,
                save_path=os.path.join(save_dir, f"{group}_components.png"),
                title=f"{group.capitalize()} Component Losses"
            )
    print('[INFO] Finished Plotting Loss Groups')