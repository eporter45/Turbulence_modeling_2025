# -*- coding: utf-8 -*-
"""
LossWeights Management and Utility Functions

- Maps output features to indices.
- Computes and manages dynamic loss weights for groups, cases, and components.
- Supports phase-switching of loss weighting during training.
- Provides plotting utility for component weight histories.
- Provides functions to sum losses within groups and overall.

Handles configuration-driven weight enabling, dynamic updating,
and weighted application of losses during training.

Created on Wed Jul  2 11:27:08 2025

@author: eoporter
"""
import os
import torch 

# --- NEW: family helpers ---
RST6 = ['uu','uv','vv','uw','vw','ww']
AIJ6 = ['a_xx','a_xy','a_yy','a_xz','a_yz','a_zz']
BIJ6 = ['b_xx','b_xy','b_yy','b_xz','b_yz','b_zz']

def detect_family(output_features):
    names = [n.lower() for n in output_features]
    if any(n.startswith('a_') for n in names):  # a_xx ...
        return 'aij'
    if any(n.startswith('b_') for n in names):  # b_xx ...
        return 'bij'
    return 'rst'

def has_tke(output_features):
    names = [n.lower() for n in output_features]
    return 'tke' in names or 'k' in names

def tke_index(output_features):
    names = [n.lower() for n in output_features]
    return names.index('tke') if 'tke' in names else names.index('k')

def get_indices(targets, output_features):
    """
    Map target feature names to their indices in the model output vector.
    """
    indices = []
    for feature in targets:
        try:
            idx = output_features.index(feature)
            indices.append(idx)
        except ValueError:
            raise ValueError(f"Feature '{feature}' not found in output features.")
    return indices


class LossWeights:
    def __init__(self, config, case_names, y_train, verbose=True):
        self.verbose = verbose
        self.config = config
        self.case_names = case_names
        self.output_features = config['features']['output']
        self.family = detect_family(self.output_features)
    
        # --- weights cfg + safe defaults ---
        weights_cfg = (config.get('training', {})
                             .get('loss', {})
                             .get('weights', {}))
        self.weights_enabled = weights_cfg.get('enabled', False)
    
        terms_cfg = config.get('training', {}).get('loss', {}).get('terms', {})
        self.phys_loss_enabled       = terms_cfg.get('phys', {}).get('enabled', True)
        self.data_loss_enabled       = terms_cfg.get('data', {}).get('enabled', True)
        self.constraint_loss_enabled = terms_cfg.get('constraint', {}).get('enabled', True)
    
        # Group weights
        self.group_cfg    = weights_cfg.get('group', {})
        self.group_enabled = self.group_cfg.get('enabled', False)
        self.group_dynamic = self.group_cfg.get('dynamic', False)
        self.group_weights = {
            'data':       self.group_cfg.get('data', 1.0),
            'phys':       self.group_cfg.get('phys', 1.0),
            'constraint': self.group_cfg.get('constraint', 1.0),
        }
    
        # Case weights  (key is 'case' in your configs)
        self.case_cfg    = weights_cfg.get('case', {})
        self.case_enabled = self.case_cfg.get('enabled', False)
        self.case_dynamic = self.case_cfg.get('dynamic', False)
    
        # Component weights (data-based)
        self.data_cfg    = weights_cfg.get('data', {})
        self.data_enabled = self.data_cfg.get('enabled', False)
        self.data_dynamic = self.data_cfg.get('dynamic', False)
    
        # Scalars derived from data_cfg
        self.tke_scalar = float(self.data_cfg.get('tke_scalar', 1.0)) if isinstance(self.data_cfg, dict) else 1.0
        self.data_eps   = float(self.data_cfg.get('epsilon', 1e-8))   if isinstance(self.data_cfg, dict) else 1e-8
    
        # Precompute weights
        self.component_weights = (
            self.compute_component_weights(y_train)
            if (self.data_enabled and y_train is not None) else
            {name: 1.0 for name in self.output_features}
        )
        self.case_weights = (
            self.compute_case_weights(y_train)
            if (self.case_enabled and y_train is not None) else
            {name: 1.0 for name in self.case_names}
        )
    
        # Master dict (what scale_losses_by_weights.apply_* expects)
        self.dict = {
            'group': self.group_weights,
            'case': self.case_weights,
            'component': self.component_weights
        }
    
        # Histories
        self.prev_component_weights = {}
        self.component_weight_history = [(0, self.component_weights.copy())]
    
        # Phase switching
        phase_cfg = weights_cfg.get('phase_switch', {})
        self.phase_enabled = phase_cfg.get('enabled', False)
        self.loss_key      = phase_cfg.get('loss_key', '')
        self.threshold     = phase_cfg.get('threshold_percent', 0.1)
        self.pre_wt        = phase_cfg.get('pre_switch_weight', 0.01)
        self.post_wt       = phase_cfg.get('post_switch_weight', 1.0)
        self.switch_triggered = False
        self.baseline_loss = None
    
        if self.phase_enabled:
            self.transition_map = self.get_transition_map(self.loss_key)
            # initialize certain components low
            for k in self.transition_map.get("increase_from_pre", []):
                if k in self.component_weights:
                    self.component_weights[k] = self.pre_wt
            self.dict['component'] = self.component_weights
    
        # Global/dynamic alphas
        self.group_alpha = 1.0
        self.dynamic_component_alpha = {name: 1.0 for name in self.output_features}

            
    def print_weights(self):
        print("[DEBUG][LossWeights] Current weights:")
        print(f"  Group weights: {self.group_weights}")
        print(f"  Case weights: {self.case_weights}")
        print(f"  Component weights: {self.component_weights}")
        print(f"  Dynamic component alpha: {self.dynamic_component_alpha}")
        print(f"  Group alpha: {self.group_alpha}")

    def compute_component_weights(self, y_train):
        """
        RST: inverse-mean/log smoothing for all outputs.
        AIJ: same for the six a_ij channels; TKE gets a scalar (tke_scalar).
        BIJ: for now, treat like RST (all outputs).
        """
        epsilon = self.data_eps
        feats = self.output_features
        fam = self.family
    
        all_labels = torch.cat(y_train, dim=0)  # [N, C]
        weights = {}
    
        if fam == 'aij':
            # Use canonical AIJ6 order for stability
            aij_names = [nm for nm in AIJ6 if nm in feats]
            idx = [feats.index(n) for n in aij_names]
            mean_components = torch.mean(torch.abs(all_labels[:, idx]), dim=0)  # [len(aij_names)]
            max_mean = mean_components.max()
        
            for mean_val, name in zip(mean_components, aij_names):
                if mean_val.item() < epsilon:
                    weights[name] = 1.5
                else:
                    raw = max_mean / (mean_val + epsilon)
                    weights[name] = (torch.log(raw + epsilon) + 1.0).item()
        
            # TKE scalar if present
            low = [n.lower() for n in feats]
            if 'tke' in low or 'k' in low:
                k_name = feats[low.index('tke')] if 'tke' in low else feats[low.index('k')]
                weights[k_name] = float(self.tke_scalar)
        
            # Default any other outputs to 1.0
            for name in feats:
                weights.setdefault(name, 1.0)
            return weights

    
        # default: RST/BIJ treat all columns equally
        mean_components = torch.mean(torch.abs(all_labels), dim=0)  # [C]
        max_mean = mean_components.max()
        for mean_val, name in zip(mean_components, feats):
            if mean_val.item() < epsilon:
                weights[name] = 1.5
            else:
                raw = max_mean / (mean_val + epsilon)
                weights[name] = (torch.log(raw + epsilon) + 1.0).item()
        return weights


    def compute_case_weights(self, y_train):
        """
        Case weights reflect the *scale* of turbulence.
        - RST: compute k from uu,vv,ww.
        - AIJ: read k directly from the 'tke' (or 'k') column.
        - BIJ: (later) could use ||b||_F or default 1.0 per case.
        """
        epsilon = 1e-8
        feats = self.output_features
        fam = self.family
        case_means = []
    
        for idx, name in enumerate(self.case_names):
            label = y_train[idx]  # [Ni, C]
            if fam == 'aij' and has_tke(feats):
                kcol = tke_index(feats)
                k_vals = label[:, kcol]
                mean_k = k_vals.mean().abs().item()
            else:
                # default: RST path
                try:
                    uu = label[:, feats.index('uu')]
                    vv = label[:, feats.index('vv')]
                    ww = label[:, feats.index('ww')]
                    mean_k = (0.5 * (uu + vv + ww)).mean().abs().item()
                except ValueError:
                    # If uu/vv/ww missing (e.g., bij family now), fall back to 1.0
                    mean_k = 1.0
            case_means.append(mean_k)
    
        mean_tensor = torch.tensor(case_means)
        max_mean = mean_tensor.max()
        weights_tensor = max_mean / (mean_tensor + epsilon)
    
        case_weights = {name: weights_tensor[i].item() for i, name in enumerate(self.case_names)}
        smoothed = {nm: (torch.log(torch.tensor(val) + epsilon).item() + 1.0)
                    for nm, val in case_weights.items()}
        return smoothed



    def get_transition_map(self, loss_key):
        if loss_key == "invariants":
            return {
                "reduce_to_zero": [
                    "I1_a_comp", "I1_a_eig", "I2_a_comp", "I2_a_eig", "I3_a_comp", "I3_a_eig",
                    "I1_b_comp", "I1_b_eig", "I2_b_comp", "I2_b_eig", "I3_b_comp", "I3_b_eig",
                    "a11", "a12", "a13", "a22", "a23", "a33",
                    "b11", "b12", "b13", "b22", "b23", "b33"
                ],
                "increase_from_pre": [
                    "I2_a_consistency", "I3_a_consistency",
                    "I2_b_consistency", "I3_b_consistency",
                    "frob_a", "frob_b",
                ]
            }
        else:
            raise ValueError(f"No phase transition map defined for loss_key: '{loss_key}'")
    
    

    
    def apply(self, loss_dict, case_name):
        from train_utils.scale_losses_by_weights import apply_weights_to_loss_dict
        if case_name not in self.case_weights:
            raise KeyError(f"[LossWeights] Case '{case_name}' not found in case_weights.")
        # Apply dynamic alphas if enabled
        group_scaled = {
            k: v * self.group_alpha if self.group_dynamic else v
            for k, v in self.group_weights.items()
        }
    
        component_scaled = {
            k: v * self.dynamic_component_alpha.get(k, 1.0)
            if self.data_dynamic else v
            for k, v in self.component_weights.items()
        }
    
        weights_dict = {
            'group': group_scaled,
            'case': self.case_weights,
            'component': component_scaled
        }

        #print(f"[DEBUG][apply] Case '{case_name}' - Original losses:")
        #print(loss_dict)
        #print(f"[DEBUG][apply] Case '{case_name}' - Weighted losses:")
        #print(weighted_losses)

        return apply_weights_to_loss_dict(loss_dict, weights_dict, case_name)
    
    def update_component_weights(self, component_losses: dict, epoch: int, lr: float = 0.1):
        """
        Updates component weights using exponential moving average with current component losses.
        Ignores any extra keys like 'net'.
        """
        # Filter incoming losses to known components
        filtered_losses = {k: v for k, v in component_losses.items() if k in self.component_weights}
    
        missing = [k for k in self.component_weights.keys() if k not in filtered_losses]
        if missing and self.verbose:
            print(f"[WARN][LossWeights] Missing component losses for: {missing}")
    
        # Save prev
        self.prev_component_weights = self.component_weights.copy()
    
        updated = {}
        for name, old_w in self.component_weights.items():
            cur_l = filtered_losses.get(name, old_w)  # fallback to old weight if missing
            updated[name] = (1 - lr) * old_w + lr * float(cur_l)
    
        self.component_weights = updated
        self.dict['component'] = updated
        self.component_weight_history.append((epoch, updated.copy()))
        if self.verbose:
            print(f'[INFO] Weights updated from: {self.prev_component_weights} to {updated}')

    def check_and_apply_phase_switch(self, epoch, loss_dict_unweighted):
        if not self.phase_enabled or self.switch_triggered:
            return
    
        if self.loss_key not in loss_dict_unweighted:
            if self.verbose:
                print(f"[PHASE] Loss key '{self.loss_key}' not found in unweighted loss dict.")
            return
    
        # Capture baseline on first epoch
        if self.baseline_loss is None:
            self.baseline_loss = sum(
                loss_dict_unweighted[self.loss_key].get(k, 0.0)
                for k in self.transition_map.get("reduce_to_zero", [])
            ) + 1e-8
            if self.verbose:
                print(f"[PHASE] Baseline {self.loss_key} loss: {self.baseline_loss:.4f}")
            return
    
        current_loss = sum(
            loss_dict_unweighted[self.loss_key].get(k, 0.0)
            for k in self.transition_map.get("reduce_to_zero", [])
        )
        ratio = current_loss / self.baseline_loss
        if ratio < self.threshold:
            self.switch_triggered = True
            if self.verbose:
                print(f"[PHASE] Switch triggered at epoch {epoch} | ratio {ratio:.4f} < {self.threshold}")
    
            for key in self.transition_map.get("reduce_to_zero", []):
                if key in self.component_weights:
                    self.component_weights[key] = 0.0
    
            for key in self.transition_map.get("increase_from_pre", []):
                if key in self.component_weights:
                    self.component_weights[key] = self.post_wt
    
            self.dict['component'] = self.component_weights     
    
    def get_weight_history(self):
        return self.component_weight_history
    
    
def plot_weights_history(history, save_dir=None, filename='component_weight_history.png'):
    import matplotlib.pyplot as plt
    if not history:
       raise ValueError("Empty history passed to plot_all_component_weights().")
    _, first_weights = history[0]
    component_names = list(first_weights.keys())
    
    #make plot
    fig, ax = plt.subplots(figsize=(10,6))
    for name in component_names:
        epochs = [e for e, _ in history]
        weights = [w[name] for _, w in history]
        #extend stepwize plotting
        step_epochs = []
        step_weights = []
        for i in range(len(epochs)):
            step_epochs.append(epochs[i])
            step_weights.append(weights[i])
            if i < len(epochs) - 1: 
                step_epochs.append(epochs[i+1])
                step_weights.append(weights[i])
        ax.plot(step_epochs, step_weights, drawstyle='steps-post', label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Component Weight")
    ax.grid(True)
    ax.legend()
    ax.tight_layout()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        fig.savefig(save_path)
        print(f"[INFO] Saved weight plot to {save_path}")

    plt.close(fig)  # Prevent display in headless environments
    
def summate_losses(loss_group: dict) -> float:
    if loss_group is None:
        return 0.0
    return sum(loss_group.values())

def summate_loss_groups(loss_dict):
    total = 0.0
    for group in ['data', 'phys', 'constraint']:
        group_losses = loss_dict.get(group)
        if group_losses is None:
            continue  # skip disabled group

        # Sum all components except 'net' key
        group_loss = sum(val for k, val in group_losses.items() if k != 'net')
        group_losses['net'] = group_loss
        #print(f"[DEBUG][summate_loss_groups] Group '{group}' loss components: {group_losses}")
        #print(f"[DEBUG][summate_loss_groups] Group '{group}' net loss: {group_loss}")
        total += group_loss
    loss_dict['net'] = total
   # print(f"[DEBUG][summate_loss_groups] Total net loss: {total}")
    return total