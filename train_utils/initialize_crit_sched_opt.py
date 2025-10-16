# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 16:27:34 2025

@author: eoporter
"""
from torch import mean as tmean
import torch.optim.lr_scheduler as sched
import torch.nn as nn
from torch.nn import MSELoss, L1Loss
import torch.optim as optim
# criterion Class
class RelativeMSELoss(nn.Module):
    def __init__(self, eps=1e-8, reduction='mean'):
        super(RelativeMSELoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, target):
        rel_error = (pred - target) / (target + self.eps)
        rel_mse = rel_error ** 2

        if self.reduction == 'none':
            return rel_mse  # shape: (N, C)
        elif self.reduction == 'mean':
            return rel_mse.mean()
        elif self.reduction == 'sum':
            return rel_mse.sum()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


#initialize criterion funct
def initialize_criterion(config):
    reduc = 'none'
    crit = config['training']['criterion'].lower()
    if crit in ('mse', 'mean_sq', 'mean_squred'):
        return MSELoss(reduction=reduc)
    elif crit in ('mae', 'l1', 'mean_avg'):
        return L1Loss(reduction=reduc)
    elif crit in ('rel_mse', 'relative mse', 'rel mse', 'relative_mse'):
        return RelativeMSELoss(reduction=reduc)
    elif crit in ('smooth', 'smooth_L1', 'L1_smooth'):
        return nn.SmoothL1Loss(beta=config['training']['beta'], reduction=reduc)
    else:
        raise TypeError(f'Criterion type {crit} is not yet supported or misspelled.')
        
#initialize optimizer
def initialize_optimizer(config, model):
    opt = config['training']['optimizer']
    opt = opt.lower()
    lrt = config['training']['lr']
    lamb = config['training']['lambda']
    if lamb == '':
        lamb = None
    alph= config['training']['alpha']
    if opt in ('adam', 'adams'):
        return optim.Adam(model.parameters(), lr=lrt, weight_decay=lamb)
    if opt in ('wadam', 'adamw', 'weightedadam', 
               'weighted_adam', 'adam_weighted',
               'w_adam', 'adam_w'):
        return optim.AdamW(model.parameters, lr=lrt, weight_decay=lamb)
    if opt in ('rms', 'rmsprop', 'rms_prop'):
        return optim.RMSprop(model.parameters(), lr=lrt, weight_decay=lamb,
                                    alpha=alph, eps=1e-5)
    else:
        raise TypeError(f'Optimizer type, {opt} is not supported or is misspelled.')

#initialize scheduler
def initialize_scheduler(optimizer, config):
    sch_cfg = config['training']['scheduler']
    epochs = config['training']['epochs'] - sch_cfg['delay']
    s_type = sch_cfg['type'].lower()
    step_sz = sch_cfg.get("step_size", None)
    gamma = sch_cfg.get('gamma', 0.1)
    #scheduler not enebled
    if not sch_cfg['enabled']:
        return None
    # no scheduler
    if s_type in ('none', 'off', 'false', ''):
        return None
    #step LR
    if s_type in ('step', 'steplr', 'lrstep'):
        if not isinstance(step_sz, int):
            raise TypeError("'step_size' must be an int for StepLR")
        return sched.StepLR(optimizer, step_size=step_sz, gamma=gamma)
    #MultistepLR
    if s_type in ('multistep', 'steps'):
        if not (isinstance(step_sz, (list, tuple) and all(isinstance(x, int) for x in step_sz))):
            raise TypeError("'step_size' must be a list of ints for MultiStepLR")
        return sched.MultiStepLR(optimizer, milestones=list(step_sz), gamma=gamma)
    #reduce on Plateu
    if s_type in ('reduce_on_plateau', 'reduce on plateau', 'reduce'):
        return sched.ReduceLROnPlateau(optimizer, mode='min',
                                       factor=float(sch_cfg['factor']),
                                       patience=int(sch_cfg['patience']),
                                       min_lr=float(sch_cfg['min_lr']),
                                       cooldown=int(sch_cfg['cooldown']))
    if s_type in ('cos', 'cosine', 'cosine_annealing', 'cos_annealing', 'cos_anneal'):
        
        return sched.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=sch_cfg['eta_min'])
    if s_type in ('warm_cos', 'warm', 'warm_cosine_annealing', 'warm_cosine', 'wca'):
        return sched.CosineAnnealingWarmRestarts(optimizer, T_0 = sch_cfg['T_0'], T_mult = sch_cfg['T_mult'], eta_min=sch_cfg['eta_min'])
    else:
        raise TypeError(f'Scheduler type, {s_type}, not yet supported or misspelled')
        
#wrapper function that returns all 3 together in one

def initialize_crit_sched_optimizer(model, config):
    criterion = initialize_criterion(config)
    optimizer = initialize_optimizer(config, model)
    scheduler = None
    if config['training']['scheduler']:
        scheduler = initialize_scheduler(optimizer, config)
    return criterion, optimizer, scheduler

