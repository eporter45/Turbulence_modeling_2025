# -*- coding: utf-8 -*-
"""
The LossTracker class is designed to track and aggregate losses
 during training at multiple granularities — by batch, by case,
 and across the entire epoch. It manages detailed histories of 
 losses categorized into data, phys, and constraint groups for 
 each training case plus an overall total. 
 
 This facilitates flexible querying of loss histories, aiding monitoring
 and debugging of training progress across different components and cases.

Created on Tue Jul  1 11:06:55 2025

@author: eoporter
"""

import torch

import torch
from collections import defaultdict

from collections import defaultdict
import copy

class LossTracker:
    def __init__(self, case_names, config):
        self.case_names = case_names
        self.scopes = ['total'] + case_names

        # histories always store plain floats (safe for JSON / plotting)
        self.histories = defaultdict(lambda: {
            'data': defaultdict(list),
            'phys': defaultdict(list),
            'constraint': defaultdict(list),
            'net': []
        })

    def _init_epoch_storage(self):
        # storage uses torch scalars (not Python floats)
        epoch_storage = {
            scope: {
                'data': defaultdict(lambda: torch.tensor(0.0)),
                'phys': defaultdict(lambda: torch.tensor(0.0)),
                'constraint': defaultdict(lambda: torch.tensor(0.0)),
                'net': torch.tensor(0.0)
            }
            for scope in self.scopes
        }
        return epoch_storage

    def update_batch(self, epoch_storage, case_name, wLoss_dict, batch_size):
        """
        Accumulate weighted losses for each batch.
        All vals are torch scalar tensors requiring grad.
        """

        # accumulate group losses
        for group in ['data', 'phys', 'constraint']:
            group_dict = wLoss_dict.get(group)
            if group_dict is None:
                continue

            for comp, loss_val in group_dict.items():
                # accumulate as differentiable torch scalars
                epoch_storage[case_name][group][comp] += loss_val
                epoch_storage['total'][group][comp] += loss_val

        # accumulate net loss (torch scalar)
        net_val = wLoss_dict.get('net', torch.tensor(0.0))
        epoch_storage[case_name]['net'] += net_val
        epoch_storage['total']['net'] += net_val

    def finalize_case(self, epoch_storage, case_name, n_points):
        if n_points == 0:
            print(f"[WARN] finalize_case: '{case_name}' has 0 points.")
            return

        # normalize by case size
        for group in ['data', 'phys', 'constraint']:
            for comp in epoch_storage[case_name][group]:
                epoch_storage[case_name][group][comp] /= n_points

            # compute group net
            group_total = sum(
                v for k, v in epoch_storage[case_name][group].items()
                if k != 'net'
            )
            epoch_storage[case_name][group]['net'] = group_total

        # total case net
        epoch_storage[case_name]['net'] = (
            epoch_storage[case_name]['data']['net']
            + epoch_storage[case_name]['phys']['net']
            + epoch_storage[case_name]['constraint']['net']
        )

    def finalize_epoch(self, epoch_storage):
        """
        Convert accumulated torch scalars → Python floats (safe for history).
        """
        for scope in self.scopes:
            for group in ['data', 'phys', 'constraint']:
                for comp, val in epoch_storage[scope][group].items():
                    self.histories[scope][group][comp].append(float(val.detach().cpu()))
            self.histories[scope]['net'].append(float(epoch_storage[scope]['net'].detach().cpu()))

    def get_all_histories(self):
        import copy
        # already floats, just return deep copy
        return copy.deepcopy(self.histories)
