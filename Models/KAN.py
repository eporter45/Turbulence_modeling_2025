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
from train_utils.initialize_crit_sched_opt import initialize_crit_sched_optimizer
from train_utils.make_loss_weights import LossWeights, summate_loss_groups
from train_utils.loss_tracker import LossTracker
from train_utils.calc_data_phys_const_losses import compute_all_losses


def initialize_kan(n_in, n_out, config, device):
    def sanitize_shape(shape):
        #converts nested shap accordingly
        return [s[0] if isinstance(s,list) and len(s) == 2 and s[1] else s for s in shape]
    print("[DEBUG][Initialize_kan] Raw model shape from config:", config['model']['shape'], flush=True)
    print("[DEBUG][Initialize_kan] Raw model type from config:", type(config['model']['shape']), flush=True)
    shape= sanitize_shape(config['model']['shape'])
    shape = [n_in] + shape + [n_out]
    print('[INFO] Sanitized KAN Shap,', shape, flush=True)
    model = KAN(width=shape,
                grid=config['model'].get('spline_order', 3),
                seed= config['model'].get('seed', 42),
                grid_range= config['model'].get('grid_range', [-1.2, 1.2]), 
                #update_grid=config['model'].get('update_grid', True),
                #use_kan= config['model'].get('use_kan', True)
                )
    print('[INFO] Kan Model initialized', flush=True)
    return model

def save_model_bundle(state_dict, optimizer, directory, tag='final model'):
    path= os.path.join(directory, tag)
    os.makedirs(path, exist_ok=True)
    torch.save(state_dict, os.path.join(path,'model_dict.pth'))
    #torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer_dict.pth'))
    

def resolve_train_case_names(config):
    trial_val = config['trial_name']
    if isinstance(trial_val, dict):
        return trial_val['train']
    else:
        return TRIALS[trial_val]['train']
    
def train_kan_model(model, x_train, y_train, config, directory, device, y_frob_max, y_k_max):
    input_size = x_train[0].shpae[1]
    output_size = y_train[0].shape[1]
    #initializzes criterion and optimizer 
    criterion, optimizer, _ = initialize_crit_sched_optimizer(model, config)
    #initialize checkpoint saving
    best_epoch = 0
    best_loss = float('inf')
    best_state_dict=copy.deepcopy(model.state_dict())
    #initialize weights and loss tracker
    train_case_names = resolve_train_case_names(config)
    weights = LossWeights(config, train_case_names, y_train, verbose=True) 
    if config['training']['loss']['weights']['enabled']:
        print(f'Initial Loss weights: \n {weights.print_weights()}')
    loss_tracker = LossTracker(train_case_names, config)
    model.to(device)
    #begin training loop
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_storage = loss_tracker._init_epoch_storage()
        #loop through all train cases
        for idx, (x, y)  in enumerate(zip(x_train, y_train)):
               name = train_case_names[idx]
               dataset = torch.utils.data.TensorDataset(x.to(device), y.to(device))
               loader = torch.utils.data.DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
               n_points = 0
               #loop through batches
               for x_batch, y_batch in loader:
                   x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                   outputs = model(x_batch)
                   #make batch losses
                   loss_dict = compute_all_losses(config, outputs, y_batch, criterion,epoch=epoch, y_frob_max=y_frob_max, y_k_max=y_k_max)
                   #weight each loss
                   weighted_loss = weights.apply(loss_dict, name)
                   loss = summate_loss_groups(weighted_loss)
                   #optimizer and torch operations
                   optimizer.zero_grad()
                   print(f'[INFO] Epoch {epoch} computed loss: {loss.item()}')

                   loss.backward()
                   optimizer.step()
                   #loss tracking
                   bs = x_batch.size(0)
                   n_points += bs
                   loss_tracker.update_batch(epoch_storage, name, weighted_loss, bs)
                   
               loss_tracker.finalize_case(epoch_storage, name, n_points=n_points)
               
        loss_tracker.finalize_epoch(epoch_storage)
        
        net_loss = epoch_storage['total']['net']
        net_phys_loss = epoch_storage['total']['phys']['net'] + epoch_storage['total']['constraint']['net']
        net_data_loss = epoch_storage['total']['data']['net'] 
        net_const_loss = epoch_storage['total']['constraint']['net']

        if net_loss < best_loss:
            best_loss = net_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            best_optimizer = copy.deepcopy(optimizer.state_dict())
            
        if epoch % config['training']['eval_every'] == 0:
            print(f"Epoch [{epoch}/{config['training']['epochs']}], Net Loss: {net_loss:.3e} Data: {net_data_loss:.3e}, Phys: {net_phys_loss:.3e}, Const: {net_const_loss:.3e} ", flush=True)
            
        if epoch % 250 ==0  and epoch !=0:
            save_model_bundle(model.to(device).state_dict(), optimizer, directory, tag=f'epoch_{epoch}')
        
    #Save best model
    best_model = initialize_kan(input_size, output_size, config, device)
    best_model.load_state_dict(best_state_dict)
    save_model_bundle(best_state_dict, best_optimizer, directory, tag='best_model')
    save_model_bundle(model.state_dict(), optimizer, directory, tag='final_model')
    #stack loss dataframe
    loss_df = loss_tracker.get_all_histories()
    
    return model, loss_df, best_model, best_epoch, best_state_dict, optimizer


def train_kan_model_with_scheduler(model, x_train, y_train, config, directory, device, y_frob_max, y_k_max):
    input_size = x_train[0].shape[1]
    output_size = y_train[0].shape[1]
    #initializzes criterion and optimizer 
    #make a script to initialize both for variability
    criterion, optimizer, scheduler = initialize_crit_sched_optimizer(model, config)

    #initialize checkpoint saving
    best_epoch = 0
    best_loss = float('inf')
    best_state_dict=copy.deepcopy(model.state_dict())
    #initialize weights and loss tracker
    train_case_names = resolve_train_case_names(config)
    weights = LossWeights(config, train_case_names, y_train, verbose=True) 
    if config['training']['loss']['weights']['enabled']:
        print(f'Initial Loss weights: \n {weights.print_weights()}')
    loss_tracker = LossTracker(train_case_names, config)
    model.to(device)
    
    #begin training loop
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_storage = loss_tracker._init_epoch_storage()
        #loop through all train cases
        for idx, (x, y)  in enumerate(zip(x_train, y_train)):
               name = train_case_names[idx]
               dataset = torch.utils.data.TensorDataset(x.to(device), y.to(device))
               loader = torch.utils.data.DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
               n_points = 0
               #loop through batches
               for x_batch, y_batch in loader:
                   x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                   outputs = model(x_batch)
                   #make batch losses
                   loss_dict = compute_all_losses(config, outputs, y_batch, criterion,epoch=epoch,y_frob_max=y_frob_max, y_k_max=y_k_max)
                   #weight each loss
                   weighted_loss = weights.apply(loss_dict, name)
                   loss = summate_loss_groups(weighted_loss)
                   #optimizer and torch operations
                   optimizer.zero_grad()
                   #print(f'[INFO] Epoch {epoch} computed loss: {loss.item()}')
                   loss.backward()
                   optimizer.step()
                   #loss tracking
                   bs = x_batch.size(0)
                   n_points += bs
                   loss_tracker.update_batch(epoch_storage, name, weighted_loss, bs)
               
                   
               
               loss_tracker.finalize_case(epoch_storage, name, n_points=n_points)
              
        loss_tracker.finalize_epoch(epoch_storage)
       
        net_loss = epoch_storage['total']['net']
        net_phys_loss = epoch_storage['total']['phys']['net'] + epoch_storage['total']['constraint']['net']
        net_data_loss = epoch_storage['total']['data']['net'] 
        net_const_loss = epoch_storage['total']['constraint']['net']

        if net_loss < best_loss:
           best_loss = net_loss
           best_epoch = epoch
           best_state_dict = copy.deepcopy(model.state_dict())
           best_optimizer = copy.deepcopy(optimizer.state_dict())
            
        if epoch % config['training']['eval_every'] == 0:
            print(f"Epoch [{epoch}/{config['training']['epochs']}], Net Loss: {net_loss:.3e} Data: {net_data_loss:.3e}, Phys: {net_phys_loss:.3e}, Const: {net_const_loss:.3e} ", flush=True)
            print(f"Epoch [{epoch}/{config['training']['epochs']}], LR: {scheduler.get_last_lr()[0]:.3e}", flush=True)
        if epoch % 250 == 0 and epoch !=0:
            save_model_bundle(model.to(device).state_dict(), optimizer, directory, tag=f'epoch_{epoch}')
        #step the scheduler
        if epoch > config['training']['scheduler']['delay']:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(net_loss)
            elif(config['training']['scheduler']['type'] == 'reduce' or 
                 config['training']['scheduler']['type'] == 'reduce_lr' or 
                 config['training']['scheduler']['type'] == 'reduce_on_plateau'):
                scheduler.step(net_loss)
            else:
                scheduler.step()
                 
    #Save best model
    best_model = initialize_kan(input_size, output_size, config, device)
    best_model.load_state_dict(best_state_dict)
    save_model_bundle(best_state_dict, best_optimizer, directory, tag='best_model')
    save_model_bundle(model.state_dict(), optimizer, directory, tag='final_model')
    #stack loss dataframe
    loss_df = loss_tracker.get_all_histories()
    
    return model, loss_df, best_model, best_epoch, best_state_dict, optimizer

def runKAN_model(x_train, y_train, x_test, config, directory, device, y_frob_max, y_k_max):
    # Set seed for reproducibility
    seed = config['training'].get('seed', 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    input_size = x_train[0].shape[1]
    output_size = y_train[0].shape[1]

    # Initialize model
    model = initialize_kan(n_in=input_size,
                           n_out=output_size,
                           config=config,
                           device=device)

    print(f'[MODEL] KAN Architecture Shape: {config["model"]["shape"]}')
    print(f'[MODEL] Grid Range: {config["model"].get("grid_range", [-1.2, 1.2])}')
    print(f'[MODEL] Spline Order: {config["model"].get("spline_order", 3)}')
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Number of learnable parameters: {total_params}")

    # Train with or without scheduler
    if config['training']['scheduler']['enabled']:
        fin_model, loss_df, best_model, best_epoch, best_state_dict, optimizer = train_kan_model_with_scheduler(
            model, x_train, y_train, config, directory, device, y_frob_max=y_frob_max, y_k_max=y_k_max
        )
    else:
        fin_model, loss_df, best_model, best_epoch, best_state_dict, optimizer = train_kan_model(
            model, x_train, y_train, config, directory, device, y_frob_max=y_frob_max, y_k_max=y_k_max
        )

    # Make predictions
    best_model.to(device)
    best_model.eval()
    with torch.no_grad():
        predictions = [best_model(x.to(device)).detach().cpu() for x in x_test]
    model.to(device)
    model.eval()
    with torch.no_grad():
        f_preds = [fin_model(x.to(device)).detach().cpu() for x in x_test]
    return f_preds, predictions, loss_df, best_model, best_epoch, best_state_dict, optimizer
