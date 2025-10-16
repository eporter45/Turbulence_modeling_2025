"""
author: @eoporter
Fully Connected Network (FCN) model and training utilities.

This script:
- Defines a configurable FCN architecture with customizable layers, width, dropout, and activations
- Trains the model using physics-informed losses, weighted by case
- Supports optional learning rate scheduler integration
- Logs loss histories and saves best and final model checkpoints
- Returns predictions for test cases

Configuration (`config`) controls model architecture, training parameters, optimizer, and loss setup.
"""





import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import copy 
from Trials import TRIALS, debug_trials
#from training_utils.weight_case_losses import build_case_losses
from train_utils.initialize_crit_sched_opt import initialize_crit_sched_optimizer
from train_utils.make_loss_weights import LossWeights, summate_loss_groups
from train_utils.loss_tracker import LossTracker
from train_utils.calc_data_phys_const_losses import compute_all_losses

def get_activation(name, neg_slope=0.01) -> nn.Module:
    name = name.lower()
    if name in ('relu6','relu'):
        return nn.ReLU()
    elif name == 'leakyrelu':
        return nn.LeakyReLU(negative_slope=neg_slope)
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name in ('linear', 'identity', 'none'):
        return nn.Identity()
    else:
        raise ValueError(f'Unsupported activation: {name}')
        

class FCN(nn.Module):
   def __init__(self, dropout, input_size, output_size, activation='leakyrelu', layers=10, width=10, neg_slope=0.01):
        super(FCN, self).__init__()
        # params
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.dropout = float(dropout)
        self.num_layers = int(layers)
        self.width = int(width)
        self.activation = activation
        self.neg_slope = float(neg_slope)
        self.activation_fn = get_activation(activation, neg_slope)
        
        layers_list = []
        # First layer: input_size -> width
        layers_list.append(nn.Linear(self.input_size, self.width))
        layers_list.append(self.activation_fn)
        layers_list.append(nn.Dropout(self.dropout))
        # Hidden layers: width -> width
        for _ in range(self.num_layers - 2):
            layers_list.append(nn.Linear(self.width, self.width))
            layers_list.append(self.activation_fn)
            layers_list.append(nn.Dropout(self.dropout))
        # Output layer: width -> output_size (no activation, no dropout)
        layers_list.append(nn.Linear(self.width, self.output_size))
        self.net = nn.Sequential(*layers_list)
        #proper initialization per activation:
        for m in self.net:
            if isinstance(m, nn.Linear):
                if self.activation == 'leakyrelu':
                    nn.init.kaiming_normal_(m.weight, a=self.neg_slope, nonlinearity = 'leaky_relu')
                elif self.activation in {'tanh', 'sigmoid'}:
                    nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain(self.activation))
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
   def forward(self, x):
        return self.net(x)
    
def compile_loss_logs(net_case_losses, phys_case_losses, data_case_losses, net_data_history, net_phys_history, net_loss_history):
    return {'case_net': net_case_losses, 
            'case_phys': phys_case_losses,
            'case_data': data_case_losses,
            'total_data': net_data_history,
            'total_phys': net_phys_history,
            'total_loss': net_loss_history}

def create_loss_history_dict(case_names):
    return ({name:[] for name in case_names},
            {name:[] for name in case_names},
            {name:[] for name in case_names})

def resolve_train_case_names(config, debug=False):
    trial_val = config['trial_name']
    if isinstance(trial_val, dict):
        return trial_val['train']
    elif debug:
        return debug_trials[trial_val]['train']
    else:
        return TRIALS[trial_val]['train']
    
def save_model_bundle(model, optimizer, directory, tag='final_model'):
    path = os.path.join(directory, tag)
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, 'model_dict.pth'))
    torch.save(optimizer, os.path.join(path, 'optimizer_dict.pth'))

def make_predictions(model, x_test):
    model.eval()
    with torch.no_grad():
        preds = [model(x) for x in x_test]
    return preds

def train_model(model, x_train, y_train, config, directory, device, y_frob_max, y_k_max):
    criterion, optimizer, _ = initialize_crit_sched_optimizer(model, config)
    best_epoch = 0
    best_loss = float('inf')
    best_model = copy.deepcopy(model)
    
    train_case_names = resolve_train_case_names(config, config['debug'])
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
                   loss_dict = compute_all_losses(config, outputs, y_batch, criterion, epoch=epoch,y_frob_max=y_frob_max, y_k_max=y_k_max)
                   #weight each loss
                   weighted_loss = weights.apply(loss_dict, name)
                   loss = summate_loss_groups(weighted_loss)
                   #optimizer and torch operations
                   optimizer.zero_grad()
                  # print(f'[INFO] Epoch {epoch} computed loss: {loss.item()}')
                   loss.backward()
                   optimizer.step()
                   #loss tracking
                   bs = x_batch.size(0)
                   n_points += bs
                   loss_tracker.update_batch(epoch_storage, name, weighted_loss, bs)
                   #print(f'[DEBUG] finished updating batch')
                   
                   
               loss_tracker.finalize_case(epoch_storage, name, n_points=n_points)
              # print(f"[DEBUG] Case: {name}, total points this epoch: {n_points}")

        loss_tracker.finalize_epoch(epoch_storage)
        
        net_loss = epoch_storage['total']['net']
        net_phys_loss = epoch_storage['total']['phys']['net']
        net_const_loss = epoch_storage['total']['constraint']['net']
        net_data_loss = epoch_storage['total']['data']['net'] 
        
        if net_loss < best_loss:
            best_model = copy.deepcopy(model)
            best_loss = net_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            best_optimizer = copy.deepcopy(optimizer.state_dict())
            
        if epoch % config['training']['eval_every'] == 0:
            print(f"Epoch [{epoch}/{config['training']['epochs']}], Net Loss: {net_loss:.3e} Data: {net_data_loss:.3e}, Phys: {net_phys_loss:.3e}, Const: {net_const_loss:.3e} ", flush=True)
            
        if epoch % 250==0 and epoch !=0:
            save_model_bundle(model, optimizer, directory, tag=f'epoch_{epoch}')
        
    #Save best model
    save_model_bundle(best_model, best_optimizer, directory, tag='best_model')
    save_model_bundle(model, optimizer, directory, tag='final_model')
    #stack loss dataframe
    loss_df = loss_tracker.get_all_histories()
    
    return model, loss_df, best_model, best_epoch, best_state_dict, optimizer
    

def train_model_with_scheduler(model, x_train, y_train, config, directory, device, y_frob_max, y_k_max):
    criterion, optimizer, scheduler = initialize_crit_sched_optimizer(model, config)
    best_epoch = 0
    best_loss = float('inf')
    best_model = copy.deepcopy(model)
    
    train_case_names = resolve_train_case_names(config, config['debug'])
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
                   #testing nans in outputs
                   if torch.isnan(outputs).any():
                        print(f"[ERROR] NaNs in model output at epoch {epoch}")
                   if torch.isnan(y_batch).any():
                        print(f"[ERROR] NaNs in target y_batch at epoch {epoch}")
                   #make batch losses
                   loss_dict = compute_all_losses(config, outputs, y_batch, criterion,epoch=epoch, y_frob_max=y_frob_max, y_k_max=y_k_max)
                   #print(f'[DEBUG] loss dict: {loss_dict}')
                   #weight each loss
                   weighted_loss = weights.apply(loss_dict, name)
                   #print(f'[DEBUG] weighted loss dict: {weighted_loss}')
                   loss = summate_loss_groups(weighted_loss)
                  # print(f'[DEBUG] loss sum: {loss} ')
                   
                   #print(f'[LOSS INFO] batch loss: {loss}')
                   #print("y_batch min/max:", y_batch.min().item(), y_batch.max().item())
                   #print("outputs min/max:", outputs.min().item(), outputs.max().item())
                   #optimizer and torch operations

                   #print(f'[INFO] Epoch {epoch} computed loss: {loss.item()}')

                   optimizer.zero_grad()
                   loss.backward()
                   optimizer.step()
                   #loss tracking
                   bs = x_batch.size(0)
                   n_points += bs
                   loss_tracker.update_batch(epoch_storage, name, weighted_loss, bs)

                   
                   
               #update case
               loss_tracker.finalize_case(epoch_storage, name, n_points=n_points)
               #print(f"[DEBUG] Case: {name}, total points this epoch: {n_points}")

        loss_tracker.finalize_epoch(epoch_storage)
        
        net_loss = epoch_storage['total']['net']
        net_phys_loss = epoch_storage['total']['phys']['net']
        net_const_loss = epoch_storage['total']['constraint']['net']
        net_data_loss = epoch_storage['total']['data']['net'] 

               
        #update epoch
        
        if net_loss < best_loss:
            best_model = copy.deepcopy(model)
            best_loss = net_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            best_optimizer = copy.deepcopy(optimizer.state_dict())
            
        if epoch % config['training']['eval_every'] == 0:
            print(f"Epoch [{epoch}/{config['training']['epochs']}], Net Loss: {net_loss:.3e} Data: {net_data_loss:.3e}, Phys: {net_phys_loss:.3e}, Const: {net_const_loss:.3e} ", flush=True)
            print(f"Epoch [{epoch}/{config['training']['epochs']}], LR: {scheduler.get_last_lr()[0]:.3e}", flush=True)
         
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
        if epoch % 500==0 and epoch !=0:
            save_model_bundle(model, optimizer, directory, tag=f'epoch_{epoch}')
        
    #Save best model
    save_model_bundle(best_model, best_optimizer, directory, tag='best_model')
    save_model_bundle(model, optimizer, directory, tag='final_model')
    #stack loss dataframe
    loss_df = loss_tracker.get_all_histories()
    
    return model, loss_df, best_model, best_epoch, best_state_dict, optimizer


def runSimple_model(X_train, y_train, X_test, config, directory, device, y_frob_max, y_k_max):
    # Set seed for reproducibility
    model = FCN(
        dropout=config['model']['dropout'],
        input_size=X_train[0].shape[1],
        output_size=y_train[0].shape[1],
        activation=config['model']['activation'],
        layers=config['model']['layers'],
        width=config['model']['width'],
    ).to(device)

    print(f'[MODEL] Num of layers: {model.num_layers} ')
    print(f'[MODEL] Width of Layers: {model.width} ')
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Number of learnable parameters: {total_params}")

    if config['training']['scheduler']['enabled']:
        final_model, loss_df, best_model, best_epoch, best_state_dict,optimizer = train_model_with_scheduler(
            model, X_train, y_train, config, directory, device, y_frob_max, y_k_max=y_k_max
        )
    else:
        final_model, loss_df, best_model, best_epoch,best_state_dict, optimizer = train_model(
            model, X_train, y_train, config, directory, device, y_frob_max, y_k_max = y_k_max
        )
   
    # Make predictions
    predictions = make_predictions(best_model.to(device), X_test)
    fin_preds = make_predictions(final_model.to(device), X_test)
    return fin_preds, predictions, loss_df, best_model, best_epoch, best_state_dict, optimizer