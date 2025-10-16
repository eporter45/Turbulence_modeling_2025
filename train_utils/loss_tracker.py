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
from train_utils.loss_name_mapping import normalize_loss_keys  # assumed external mapping util

from collections import defaultdict
import copy

class LossTracker:
    def __init__(self, case_names, config):
        self.case_names = case_names
        self.scopes = ['total'] + case_names
        
        self.histories = defaultdict(lambda: {
            'data': defaultdict(list),
            'phys': defaultdict(list),
            'constraint': defaultdict(list),
            'net': []
            })        
    def _init_epoch_storage(self):
        epoch_storage = {scope: {'net': 0.0,
                                 'data': defaultdict(float),
                                 'phys': defaultdict(float),
                                 'constraint': defaultdict(float)}
                         for scope in self.scopes}
        return epoch_storage
    
    def update_batch(self, epoch_storage, case_name, wLoss_dict, batch_size):
        for group in ['data', 'phys', 'constraint']:
            group_dict = wLoss_dict.get(group)
            if group_dict is None:
                continue
            for comp, loss_val in group_dict.items():
                val = loss_val
                epoch_storage[case_name][group][comp] += val 
                epoch_storage['total'][group][comp] += val 
                
                '''
                case has 1010 points
                batch size is 100
                
                
                '''
    
        # ✅ Add these lines to accumulate net loss properly
        epoch_storage[case_name]['net'] += wLoss_dict.get('net', 0.0) 
        epoch_storage['total']['net'] += wLoss_dict.get('net', 0.0)  
    
    def finalize_case(self, epoch_storage, case_name, n_points):
        if n_points == 0:
            print(f"[WARNING] finalize_case: Case '{case_name}' had 0 training points — skipping loss normalization.")
            return
        #print(f"[DEBUG][finalize_case] Case '{case_name}', n_points={n_points}")
        for group in ['data', 'phys', 'constraint']:
           #print(f"  Before normalization [{group}]: {epoch_storage[case_name][group]}")
            for comp in epoch_storage[case_name][group]:
               epoch_storage[case_name][group][comp] /= n_points
            
            # Sum all components except 'net' key to get group total
            group_total = sum(val for k, val in epoch_storage[case_name][group].items() if k != 'net')
            epoch_storage[case_name][group]['net'] = group_total
            #print(f"  After normalization [{group}]: {epoch_storage[case_name][group]}")
            #print(f"  Group '{group}' net loss: {group_total:.6f}")
    
        case_data_net = epoch_storage[case_name]['data'].get('net', 0.0)
        case_phys_net = epoch_storage[case_name]['phys'].get('net', 0.0)
        case_constraint_net = epoch_storage[case_name]['constraint'].get('net', 0.0)
        case_net = case_data_net + case_phys_net + case_constraint_net
        epoch_storage[case_name]['net'] = case_net
        #print(f"  Case '{case_name}' total net loss: {case_net:.6f}")
        
    def finalize_epoch(self, epoch_storage):
        #print("[DEBUG][finalize_epoch] Finalizing epoch...")
        for scope in self.scopes:
            for group in ['data', 'phys', 'constraint']:
                for comp, val in epoch_storage[scope][group].items():
                    self.histories[scope][group][comp].append(val)
            self.histories[scope]['net'].append(epoch_storage[scope]['net'])
            #print(f"  Scope '{scope}': net loss = {epoch_storage[scope]['net']:.6f}")
        
    
    
    def get_all_histories(self):
        def recursive_to_dict(d):
            if isinstance(d, defaultdict):
                d = {k: recursive_to_dict(v) for k, v in d.items()}
            elif isinstance(d, dict):
                d = {k: recursive_to_dict(v) for k, v in d.items()}
            return d

        return recursive_to_dict(self.histories)