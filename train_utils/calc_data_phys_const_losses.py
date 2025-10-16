'''
@author eoporter
'''
import torch
import torch.nn as nn
from train_utils.helpers import (
    get_indices, extract_rst, extract_aij, extract_six,
    detect_family, has_tke, extract_aij_and_k,
    RST6, AIJ6, tke_index, reconstruct_rst
)

def frobenius_loss(R_pred, R_true):
    return torch.mean(torch.norm(R_pred - R_true, dim=(-2, -1), p='fro'))

def log_euclidean_loss(R_pred, R_true, eps=1e-12):
    """
    Log-Euclidean loss between two SPD matrix batches.
    R_pred, R_true: [...,3,3] SPD matrices
    """
    # symmetrize
    R_pred = 0.5 * (R_pred + R_pred.transpose(-1, -2))
    R_true = 0.5 * (R_true + R_true.transpose(-1, -2))

    def mat_log(R):
        eigvals, eigvecs = torch.linalg.eigh(R)
        eigvals = torch.clamp(eigvals, min=eps)
        log_eigvals = torch.log(eigvals)
        return eigvecs @ torch.diag_embed(log_eigvals) @ eigvecs.transpose(-1, -2)

    log_Rp = mat_log(R_pred)
    log_Rt = mat_log(R_true)

    return torch.mean(torch.norm(log_Rp - log_Rt, dim=(-2, -1), p='fro'))


def spd_loss(R_pred):
    #loss on sym pos def matrices (pos eigenvalues)
    eigvals = torch.linalg.eigvalsh(R_pred)
    return torch.mean(torch.relu(-eigvals))  # penalty for negative eigenvalues

def riemannian_distance(R1, R2, eps=1e-12):
    """
    Affine-invariant Riemannian distance between SPD matrices.
    R1, R2: [...,3,3] SPD matrices
    """
    # symmetrize just in case
    R1 = 0.5 * (R1 + R1.transpose(-1, -2))
    R2 = 0.5 * (R2 + R2.transpose(-1, -2))

    # Inverse sqrt of R1
    eigvals, eigvecs = torch.linalg.eigh(R1)
    eigvals = torch.clamp(eigvals, min=eps)
    R1_inv_sqrt = (eigvecs @ torch.diag_embed(eigvals.rsqrt()) @ eigvecs.transpose(-1, -2))

    # Transform R2 into R1's basis
    M = R1_inv_sqrt @ R2 @ R1_inv_sqrt

    # Log of eigenvalues
    lam = torch.linalg.eigvalsh(M)
    lam = torch.clamp(lam, min=eps)
    log_lam = torch.log(lam)

    # Frobenius norm of log eigenvalues
    dist = torch.norm(log_lam, dim=-1)  # per-sample
    return dist.mean()



def compute_k(r_ij, output_features):
    """
    Compute turbulent kinetic energy k from the normal stress components.
    """
    idxs = get_indices(['uu', 'vv', 'ww'], output_features)
    r_ii = r_ij[:, idxs]  # shape: (batch, 3)
    k = 0.5 * r_ii.sum(dim=1)  # shape: (batch,)
    return k


def compute_aij(r_ij, tke, output_features):
    """
    Compute anisotropy tensor a_ij from RST and k.
    """
    a_ij = torch.zeros_like(r_ij)  # shape: (batch, C)
    two_thirds_k = (2.0 / 3.0) * tke.unsqueeze(1)  # shape: (batch, 1)
    for i, name in enumerate(output_features):
        if name in ['uu', 'vv', 'ww']:
            a_ij[:, i] = r_ij[:, i] - two_thirds_k.squeeze(1)
        else:
            a_ij[:, i] = r_ij[:, i]
    return a_ij


def compute_bij(r_ij, tke, output_features):
    """
    Compute normalized anisotropy tensor b_ij from a_ij and k.
    """
    two_k = 2.0 * tke.view(-1, 1)
    b_ij =r_ij/two_k #shape: (batch, C)
    
    # Subtract 1/3 from diagonals
    for i, name in enumerate(output_features):
        if name in ['uu', 'vv', 'ww']:
            b_ij[:, i] -= 1.0 / 3.0

    return b_ij

def compute_invariants_tensor(tensor_batch, output_features=None):
    """
    Assumes tensor_batch is [N,6] in canonical order:
    [xx, xy, yy, xz, yz, zz] mapped to matrix
        [[xx, xy, xz],
         [xy, yy, yz],
         [xz, yz, zz]]
    Works for RST, a_ij, or b_ij as long as the vector is in that order.
    """
    a = tensor_batch
    xx, xy, yy, xz, yz, zz = a[:,0], a[:,1], a[:,2], a[:,3], a[:,4], a[:,5]
    I1 = xx + yy + zz
    # I2 = -a_ij a_ji  (for symmetric 3x3 => formula below)
    I2 = -(xx*xx + yy*yy + zz*zz + 2*(xy*xy + xz*xz + yz*yz))

    # I3 = det(A) for symmetric 3x3
    I3 = (
        xx*(yy*zz - yz*yz)
        - xy*(xy*zz - xz*yz)
        + xz*(xy*yz - xz*yy)
    )
    return I1, I2, I3

def compute_eigenvalues_batch(tensor_batch):
    """
    Converts [N,6] vec6 -> [N,3,3] using the same canonical order as above.
    """
    a = tensor_batch
    N = a.shape[0]
    mat = torch.zeros((N, 3, 3), dtype=a.dtype, device=a.device)
    mat[:,0,0] = a[:,0]  # xx
    mat[:,0,1] = mat[:,1,0] = a[:,1]  # xy
    mat[:,1,1] = a[:,2]  # yy
    mat[:,0,2] = mat[:,2,0] = a[:,3]  # xz
    mat[:,1,2] = mat[:,2,1] = a[:,4]  # yz
    mat[:,2,2] = a[:,5]  # zz
    return torch.linalg.eigvalsh(mat)

def invariants_alignment_loss(I_pred, I_true, eps=1e-12):
    """
    Alignment-style loss between invariants (scale-invariant).
    1 - cosine similarity between invariant vectors.
    """
    I_pred = torch.stack(I_pred, dim=-1)  # [N,3]
    I_true = torch.stack(I_true, dim=-1)  # [N,3]
    norm_p = torch.norm(I_pred, dim=-1, keepdim=True) + eps
    norm_t = torch.norm(I_true, dim=-1, keepdim=True) + eps
    I_pred = I_pred / norm_p
    I_true = I_true / norm_t
    cos_sim = torch.sum(I_pred * I_true, dim=-1)
    return torch.mean(1.0 - cos_sim)



from copy import deepcopy
def compute_feature_loss(pred, label, criterion, reduction='mean'):
    """
    Plain per-element feature loss using given criterion.
    """
    crit = type(criterion)(reduction='none')
    loss = crit(pred, label)  # shape: [batch, features]
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

def make_phys_losses(config, pred, truth, criterion):
    """
    Physics-aware losses:
    - If outputs are RST: guardrail with anisotropy consistency (aij Frobenius).
    - If outputs are AIJ(+k): elevate to RST and apply SPD-aware metrics.
    - If outputs are BIJ: placeholder for now.
    """
    phys_cfg = config['training']['loss']['terms']['phys']
    if not phys_cfg.get('enabled', False):
        return None
    types = [t.lower() for t in phys_cfg.get('types', [])]
    if not types:
        return None

    output_features = config['features']['output']
    family = detect_family(output_features)
    losses = {}

    # --- RST family ---
    if family == 'rst':
        t_pred, _ = extract_rst(pred, output_features)
        t_true, _ = extract_rst(truth, output_features)
        k_p = compute_k(t_pred, output_features)
        k_t = compute_k(t_true, output_features)
        if 'aij' in types or 'a' in types:
            a_p = compute_aij(t_pred, k_p, output_features)
            a_t = compute_aij(t_true, k_t, output_features)
            losses['phys_aij_frob'] = frobenius_loss(a_p, a_t)

        if 'tke' in types or 'k' in types:
            # TKE MSE loss
            crit = type(criterion)(reduction='none')
            losses['phys_tke'] = crit(k_p, k_t).mean()


    # --- AIJ (+k) family ---
    elif family == 'aij':
        a_p, k_p = extract_aij_and_k(pred, output_features)
        a_t, k_t = extract_aij_and_k(truth, output_features)

        if k_p is not None:
            R_pred, R_true = reconstruct_rst(pred, truth, output_features)

            if 'frob' in types:
                losses['phys_rst_frob'] = frobenius_loss(R_pred, R_true)
            if 'log_euclidean' in types:
                losses['rst_logeuclidean'] = log_euclidean_loss(R_pred, R_true)
            if 'riemann' in types:
                losses['rst_riemann'] = riemannian_distance(R_pred, R_true)
            if 'spd' in types:
                losses['rst_spd'] = spd_loss(R_pred)

    # --- BIJ family ---
    elif family == 'bij':
        if 'frob' in types:
            crit = type(criterion)(reduction='none')
            losses['phys_bij_frob'] = crit(pred, truth).mean()

    else:
        raise ValueError(f"Unknown family {family}")

    return losses

def make_constraint_losses(config, pred, truth, criterion, epoch=0):
    cfg = config['training']['loss']['terms'].get('constraint', {})
    if not cfg.get('enabled', False):
        return None
    types = [t.lower() for t in cfg.get('types', [])]
    if not types:
        return None

    # warmups
    warm_cfg = config['training']['loss'].get('warmup', {})
    eig_warm = int(warm_cfg.get('eigen', 0))
    inv_warm = int(warm_cfg.get('invariants', 0))

    def allow(kind: str):
        if 'eig' in kind and epoch < eig_warm:
            return False
        if kind.startswith('inv') and epoch < inv_warm:
            return False
        return True

    output_features = config['features']['output']
    family = detect_family(output_features)
    losses = {}

    def add_inv_losses(prefix, A_pred, A_true):
        I1_t, I2_t, I3_t = compute_invariants_tensor(A_true)
        I1_p, I2_p, I3_p = compute_invariants_tensor(A_pred)

        # component invariants
        if allow('inv') and f'inv_{prefix}_comp' in types:
            losses[f'I1_{prefix}_comp'] = compute_feature_loss(I1_p, I1_t, criterion, reduction='mean')
            losses[f'I2_{prefix}_comp'] = compute_feature_loss(I2_p, I2_t, criterion, reduction='mean')
            losses[f'I3_{prefix}_comp'] = compute_feature_loss(I3_p, I3_t, criterion, reduction='mean')

        # eigen invariants
        if allow('eig') and f'inv_{prefix}_eig' in types:
            eig_p = compute_eigenvalues_batch(A_pred)
            eig_t = compute_eigenvalues_batch(A_true)
            I1_eig_p = eig_p.sum(dim=1);  I2_eig_p = -torch.sum(eig_p**2, dim=1);  I3_eig_p = torch.prod(eig_p, dim=1)
            I1_eig_t = eig_t.sum(dim=1);  I2_eig_t = -torch.sum(eig_t**2, dim=1);  I3_eig_t = torch.prod(eig_t, dim=1)

            losses[f'I1_{prefix}_eig'] = compute_feature_loss(I1_eig_p, I1_eig_t, criterion, reduction='mean')
            losses[f'I2_{prefix}_eig'] = compute_feature_loss(I2_eig_p, I2_eig_t, criterion, reduction='mean')
            losses[f'I3_{prefix}_eig'] = compute_feature_loss(I3_eig_p, I3_eig_t, criterion, reduction='mean')

        # consistency between component and eigen invariants
        if allow('eig') and 'inv_consistency' in types:
            losses[f'I2_{prefix}_consistency'] = compute_feature_loss(I2_p, I2_eig_p, criterion, reduction='mean')
            losses[f'I3_{prefix}_consistency'] = compute_feature_loss(I3_p, I3_eig_p, criterion, reduction='mean')

        # alignment (scale-invariant)
        if allow('inv') and 'inv_align' in types:
            losses[f'I_align_{prefix}'] = invariants_alignment_loss((I1_p, I2_p, I3_p),
                                                                   (I1_t, I2_t, I3_t))

    if family == 'rst':
        t_p, _ = extract_rst(pred, output_features)
        t_t, _ = extract_rst(truth, output_features)

        k_p = compute_k(t_p, output_features)
        k_t = compute_k(t_t, output_features)

        a_p = compute_aij(t_p, k_p, output_features)
        b_p = compute_bij(t_p, k_p, output_features)
        a_t = compute_aij(t_t, k_t, output_features)
        b_t = compute_bij(t_t, k_t, output_features)

        if any('a' in k for k in types):
            add_inv_losses('a', a_p, a_t)
        if any('b' in k for k in types):
            add_inv_losses('b', b_p, b_t)


    elif family == 'aij':
        a_p, k_p = extract_aij_and_k(pred, output_features)
        a_t, k_t = extract_aij_and_k(truth, output_features)

        add_inv_losses('a', a_p, a_t)

        if k_p is not None:
            two_k_p = 2.0 * k_p.view(-1,1);  two_k_t = 2.0 * k_t.view(-1,1)
            b_p = a_p / (two_k_p + 1e-12);  b_t = a_t / (two_k_t + 1e-15)
            for i in [0,2,5]:
                b_p[:, i] -= 1.0/3.0;  b_t[:, i] -= 1.0/3.0
            add_inv_losses('b', b_p, b_t)

    elif family == 'bij':
        # later: reconstruct and apply same invariant checks
        return {}

    else:
        raise ValueError(f"Unknown family {family}")

    return losses


def make_data_losses(config, pred, truth, criterion=None):
    """
    Data losses:
    - RST outputs: SPD-aware metrics (frob, log-euclidean, riemann, spd).
    - AIJ(+k) outputs: component-wise crit/frob in anisotropy space.
    - BIJ outputs: component-wise crit/frob in normalized anisotropy space.
    """
    term_cfg = config['training']['loss']['terms']['data']
    if not term_cfg.get('enabled', False):
        return None
    types = [t.lower() for t in term_cfg.get('types', [])]
    if not types:
        return None

    output_features = config['features']['output']
    family = detect_family(output_features)
    losses = {}

    # --- RST family ---
    if family == 'rst':
        R_pred, R_true = reconstruct_rst(pred, truth, output_features)
        if 'comp' in types:
            # loop over each RST component
            for i, feat in enumerate(output_features):
                losses[f"data_comp_{feat}"] = criterion(pred[:, i], truth[:, i])
        if 'crit' in types:
            losses['crit'] = compute_feature_loss(R_pred, R_pred, criterion)
        if 'frob' in types:
            losses['data_frob'] = frobenius_loss(R_pred, R_true)
        if 'log_euclidean' in types:
            losses['data_logeuclidean'] = log_euclidean_loss(R_pred, R_true)
        if 'spd' in types:
            losses['data_spd'] = spd_loss(R_pred)
        if 'riemann' in types:
            losses['data_riemann'] = riemannian_distance(R_pred, R_true)

    elif family == 'aij':
        a_p, k_p = extract_aij_and_k(pred, output_features)
        a_t, k_t = extract_aij_and_k(truth, output_features)
        if len(types) != 1:
            raise ValueError(f"Only one data loss supported, entered {types}")
        if 'comp' in types:
            # loop over each RST component
            for i, feat in enumerate(output_features):
                losses[feat] = compute_feature_loss(pred[:, i], truth[:, i], criterion, reduction='mean')
        elif 'crit' in types and 'frob' not in types:
            # Case 1: crit on both aij and k
            crit = type(criterion)(reduction='none')
            if k_p is not None:
                losses['data_k_crit'] = crit(k_p, k_t).mean()
            per_elem = crit(a_p, a_t)  # [batch,6]
            losses['data_aij_crit'] = per_elem.mean()

        elif 'frob' in types and 'crit' not in types:
            # Case 2: frob on aij, crit on k
            if k_p is not None:
                crit = type(criterion)(reduction='none')
                losses['data_k_crit'] = crit(k_p, k_t).mean()
            #R_a_p, R_a_t = reconstruct_rst(a_p, a_t, AIJ6)
            losses['data_aij_frob'] = frobenius_loss(a_p, a_t)

        else:
            raise ValueError("For AIJ family, loss types must be either ['crit'] or ['frob'], not both.")


    # --- BIJ family ---
    elif family == 'bij':
        if 'crit' in types:
            crit = type(criterion)(reduction='none')
            losses['data_crit'] = crit(pred, truth).mean()
        if 'frob' in types:
            crit = type(criterion)(reduction='none')
            losses['data_frob'] = crit(pred, truth).mean()

    else:
        raise ValueError(f"Unknown family {family}")

    return losses


def compute_all_losses(config, pred, truth, criterion, y_frob_max, y_k_max, epoch=0):
    """
    Family-aware de-normalization before computing data/phys/constraint losses.
    - RST/BIJ: scale all outputs by y_frob_max (legacy behavior).
    - AIJ(+k): scale a_ij components by y_frob_max and k/tke by y_k_max (if provided).
    """
    losses = {}

    feats   = config['features']['output']
    family  = detect_family(feats)

    # work on clones to avoid side-effects upstream
    pred_d  = pred.clone()
    truth_d = truth.clone()

    # use .get to avoid KeyError if someone forgets to put the switch in cfg
    if config['features'].get('denorm_loss', False):
        if family == 'aij':
            # scale the six anisotropy channels
            a_idx = get_indices(AIJ6, feats)
            pred_d[:, a_idx]  *= y_frob_max
            truth_d[:, a_idx] *= y_frob_max

            # scale k/tke independently if present and scale provided
            if has_tke(feats) and (y_k_max is not None):
                ki = tke_index(feats)
                pred_d[:, ki]  *= y_k_max
                truth_d[:, ki] *= y_k_max
        else:
            # legacy path: RST/BIJ or anything else → single scaler
            pred_d  *= y_frob_max
            truth_d *= y_frob_max

    # === group losses ===
    data_cfg = config['training']['loss']['terms'].get('data', {})
    losses['data'] = make_data_losses(config, pred_d, truth_d, criterion) if data_cfg.get('enabled', False) else None

    phys_cfg = config['training']['loss']['terms'].get('phys', {})
    losses['phys'] = make_phys_losses(config, pred_d, truth_d, criterion) if phys_cfg.get('enabled', False) else None

    const_cfg = config['training']['loss']['terms'].get('constraint', {})
    losses['constraint'] = make_constraint_losses(config, pred_d, truth_d, criterion, epoch=epoch) if const_cfg.get('enabled', False) else None

    return losses

