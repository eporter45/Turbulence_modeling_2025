###############################################################
# --- Family & indexing utilities ---
###############################################################

RST6 = ['uu','uv','vv','uw','vw','ww']
AIJ6 = ['a_xx','a_xy','a_yy','a_xz','a_yz','a_zz']
BIJ6 = ['b_xx','b_xy','b_yy','b_xz','b_yz','b_zz']

def detect_family(output_features):
    names = [n.lower() for n in output_features]
    if any(n.startswith('a_') for n in names):
        return 'aij'
    if any(n.startswith('b_') for n in names):
        return 'bij'
    return 'rst'

def get_indices(targets, output_features):
    indices = []
    for feature in targets:
        try:
            idx = output_features.index(feature)
            indices.append(idx)
        except ValueError:
            raise ValueError(f"Feature '{feature}' not found in output features.")
    return indices

def extract_six(preds, output_features, family):
    if family == 'rst':
        names = RST6
    elif family == 'aij':
        names = AIJ6
    elif family == 'bij':
        names = BIJ6
    else:
        raise ValueError(f"Unknown family {family}")
    idxs = get_indices(names, output_features)
    return preds[:, idxs], names

def extract_rst(preds, output_features):
    indices = get_indices(RST6, output_features)
    return preds[:, indices], [output_features[i] for i in indices]

def extract_aij(preds, output_features):
    indices = get_indices(AIJ6, output_features)
    return preds[:, indices], [output_features[i] for i in indices]

def has_tke(output_features):
    names = [n.lower() for n in output_features]
    return 'tke' in names or 'k' in names

def tke_index(output_features):
    names = [n.lower() for n in output_features]
    return names.index('tke') if 'tke' in names else names.index('k')

def extract_aij_and_k(tensor, output_features):
    a_idx = get_indices(AIJ6, output_features)
    aij = tensor[:, a_idx]
    k = None
    if has_tke(output_features):
        k = tensor[:, tke_index(output_features)]
    return aij, k

def ij_from_name(name):
    mapping = {'u': '1', 'v': '2', 'w': '3'}
    if len(name) != 2:
        raise ValueError(f"Invalid RST component name: {name}")
    return f"{mapping[name[0]]}{mapping[name[1]]}"

import torch

def vec6_to_sym3(tensor6):
    """
    Convert a [N,6] tensor [xx, xy, yy, xz, yz, zz] into [N,3,3] symmetric matrices.
    """
    xx, xy, yy, xz, yz, zz = torch.unbind(tensor6, dim=-1)
    mat = torch.stack([
        torch.stack([xx, xy, xz], dim=-1),
        torch.stack([xy, yy, yz], dim=-1),
        torch.stack([xz, yz, zz], dim=-1)
    ], dim=-2)
    return mat

def reconstruct_rst(pred, truth, output_features):
    """
    Reconstruct Reynolds stress tensors R_ij (3x3) from model outputs.
    - If outputs are RST6 → direct mapping to [3x3].
    - If outputs are AIJ6 + k/tke → build R = 2k (a_ij + δ_ij/3).
    Returns: (R_pred, R_true) with shape [N,3,3].
    """
    names = [n.lower() for n in output_features]

    # --- Case 1: direct RST6 prediction ---
    if all(f in names for f in ['uu','uv','vv','uw','vw','ww']):
        idx = [names.index(f) for f in ['uu','uv','vv','uw','vw','ww']]
        R_pred = vec6_to_sym3(pred[:, idx])
        R_true = vec6_to_sym3(truth[:, idx])
        return R_pred, R_true

    # --- Case 2: anisotropy + k prediction ---
    if all(f in names for f in ['a_xx','a_xy','a_yy','a_xz','a_yz','a_zz']) and ('tke' in names or 'k' in names):
        a_idx = [names.index(f) for f in ['a_xx','a_xy','a_yy','a_xz','a_yz','a_zz']]
        k_idx = names.index('tke') if 'tke' in names else names.index('k')
        aij_pred, aij_true = pred[:, a_idx], truth[:, a_idx]
        k_pred, k_true = pred[:, k_idx], truth[:, k_idx]

        A_pred = vec6_to_sym3(aij_pred)
        A_true = vec6_to_sym3(aij_true)

        eye = torch.eye(3, device=pred.device, dtype=pred.dtype)
        R_pred = 2.0 * k_pred[:, None, None] * (A_pred + eye/3.0)
        R_true = 2.0 * k_true[:, None, None] * (A_true + eye/3.0)
        return R_pred, R_true

    raise ValueError("Output features must be RST6 or AIJ6+K to reconstruct RST")
