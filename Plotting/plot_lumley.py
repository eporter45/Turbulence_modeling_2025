import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import sys
import math
#Dynamically resolve project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add to system path if not already
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ---------------- utils ----------------
def _ensure_dir(d):
    if d: os.makedirs(d, exist_ok=True)

def _to_numpy(x):
    if x is None: return None
    if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
    return np.asarray(x)

# ---------- generic centerline helpers ----------
def _centerline_band(df, y_col="Cy", tol_y=1e-4, frac_fallback=0.02):
    d = df.copy()
    d["abs_y"] = np.abs(d[y_col])
    sub = d.loc[d["abs_y"] <= tol_y]
    if sub.empty:
        k = max(1, int(frac_fallback * len(d)))
        sub = d.nsmallest(k, "abs_y")
    return sub

def _pick_even_in_x(sub, n, x_col="Cx"):
    sub = sub.sort_values(x_col)
    xs = sub[x_col].to_numpy()
    if len(xs) <= n:
        return sub
    x_targets = np.linspace(xs.min(), xs.max(), n)
    arr_x  = sub[x_col].to_numpy()
    arr_ix = sub.index.to_numpy()
    used = set()
    chosen = []
    for xt in x_targets:
        j = int(np.argmin(np.abs(arr_x - xt)))
        cand = j
        if arr_ix[cand] in used:
            L, R = j-1, j+1
            picked = None
            while L >= 0 or R < len(arr_x):
                if L >= 0 and arr_ix[L] not in used:
                    picked = L; break
                if R < len(arr_x) and arr_ix[R] not in used:
                    picked = R; break
                L -= 1; R += 1
            if picked is not None:
                cand = picked
        used.add(arr_ix[cand])
        chosen.append(arr_ix[cand])
    return sub.loc[chosen]


# ---------- EXP: sample centerline & compute C1,C2,C3 from uu..ww ----------
def _exp_centerline_c123(exp_df, n=20, tol_y=1e-4, x_col="Cx", y_col="Cy", eps=1e-12):
    #df = exp_df.copy()
    df = exp_df
    # pick centerline band and evenly in x
    band   = _centerline_band(df, y_col=y_col, tol_y=tol_y)
    picked = _pick_even_in_x(band, n=n, x_col=x_col)

    # pull stresses (allow uw,vw missing → zeros)
    need_min = {x_col, y_col, "uu", "uv", "vv", "ww"}
    missing  = need_min - set(df.columns)
    if missing:
        raise KeyError(f"EXP missing columns: {sorted(missing)}")

    uu = picked["uu"].to_numpy(float)
    uv = picked["uv"].to_numpy(float)
    vv = picked["vv"].to_numpy(float)
    ww = picked["ww"].to_numpy(float)
    uw = picked["uw"].to_numpy(float) if "uw" in picked.columns else np.zeros_like(uu)
    vw = picked["vw"].to_numpy(float) if "vw" in picked.columns else np.zeros_like(uu)

    # build a_ij then b_ij
    k = 0.5 * (uu + vv + ww)
    two_thirds_k = (2.0/3.0) * k
    a_xx, a_yy, a_zz = uu - two_thirds_k, vv - two_thirds_k, ww - two_thirds_k
    a_xy, a_xz, a_yz = uv, uw, vw

    m = len(picked)
    A = np.zeros((m, 3, 3), dtype=float)
    A[:,0,0] = a_xx; A[:,1,1] = a_yy; A[:,2,2] = a_zz
    A[:,0,1] = A[:,1,0] = a_xy
    A[:,0,2] = A[:,2,0] = a_xz
    A[:,1,2] = A[:,2,1] = a_yz

    denom = 2.0 * k
    safe  = np.where(np.abs(denom) < eps, 1.0, denom)
    b = A / safe[:, None, None]
    b[np.abs(denom) < eps] = 0.0

    # barycentric C1,C2,C3 from eigenvalues (sorted λ1≥λ2≥λ3)
    lam = np.linalg.eigvalsh(b)[:, ::-1]
    l1, l2, l3 = lam[:,0], lam[:,1], lam[:,2]
    C1 = (l1 - l2)
    C2 = 2.0*(l2 - l3)
    C3 = 3.0*l3 + 1.0

    # normalize so C1+C2+C3=1 and handle degenerate rows
    S = C1 + C2 + C3
    bad = np.abs(S) < eps
    S[bad] = 1.0
    C1, C2, C3 = C1/S, C2/S, C3/S
    if np.any(bad):
        C1[bad], C2[bad], C3[bad] = 0.0, 0.0, 1.0

    return {"c1": C1, "c2": C2, "c3": C3}

# ---------- RANS: sample centerline & compute C1,C2,C3 only for those rows ----------
def _rans_centerline_c123(rans_df, n=20, tol_y=1e-4, x_col="Cx", y_col="Cy", eps=1e-12):
    #df =rans_df.copy()
    df = rans_df
    need = {x_col, y_col, "uu", "uv", "vv", "ww"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"RANS missing columns: {sorted(missing)}")

    band   = _centerline_band(df, y_col=y_col, tol_y=tol_y)
    picked = _pick_even_in_x(band, n=n, x_col=x_col)

    uu = picked["uu"].to_numpy(float)
    uv = picked["uv"].to_numpy(float)
    vv = picked["vv"].to_numpy(float)
    ww = picked["ww"].to_numpy(float)
    uw = np.zeros_like(uu)  # you confirmed uw,vw are absent
    vw = np.zeros_like(uu)

    k = 0.5 * (uu + vv + ww)
    two_thirds_k = (2.0/3.0) * k
    a_xx = uu - two_thirds_k
    a_yy = vv - two_thirds_k
    a_zz = ww - two_thirds_k
    a_xy, a_xz, a_yz = uv, uw, vw

    m = len(picked)
    A = np.zeros((m, 3, 3), dtype=float)
    A[:,0,0] = a_xx; A[:,1,1] = a_yy; A[:,2,2] = a_zz
    A[:,0,1] = A[:,1,0] = a_xy
    A[:,0,2] = A[:,2,0] = a_xz
    A[:,1,2] = A[:,2,1] = a_yz

    denom = 2.0 * k
    safe = np.where(np.abs(denom) < eps, 1.0, denom)
    b = A / safe[:, None, None]
    b[np.abs(denom) < eps] = 0.0

    lam = np.linalg.eigvalsh(b)[:, ::-1]
    l1, l2, l3 = lam[:,0], lam[:,1], lam[:,2]
    C1 = (l1 - l2)
    C2 = 2.0*(l2 - l3)
    C3 = 3.0*l3 + 1.0

    S = C1 + C2 + C3
    bad = np.abs(S) < eps
    S[bad] = 1.0
    C1 /= S; C2 /= S; C3 /= S
    if np.any(bad):
        C1[bad], C2[bad], C3[bad] = 0.0, 0.0, 1.0

    return {"c1": C1, "c2": C2, "c3": C3}

# ---------- optional preds (already coords dict) ----------
def bcoords_from_pred(pred_coords_dict):
    if pred_coords_dict is None: return None
    for k in ("c1","c2","c3"):
        if k not in pred_coords_dict: raise KeyError(f"pred coords missing '{k}'")
    return pred_coords_dict


def plot_lumley(rans_path, exp_path, y_pred=None, save_dir=None, title=None,
                n=20, tol_y=1e-4, x_col="Cx", y_col="Cy"):
    def _triangle_vertices():
        V1 = np.array([1.0, 0.0])                 # 1C
        V2 = np.array([0.0, 0.0])                 # 2C
        V3 = np.array([0.5, np.sqrt(3.0)/2.0])    # 3C
        return V1, V2, V3
    def _xy_from_c123(C1, C2, C3):
        V1, V2, V3 = _triangle_vertices()
        S = C1 + C2 + C3
        S = np.where(np.abs(S) < 1e-12, 1.0, S)
        c1, c2, c3 = C1/S, C2/S, C3/S
        x = c1*V1[0] + c2*V2[0] + c3*V3[0]
        y = c1*V1[1] + c2*V2[1] + c3*V3[1]
        return x, y
    print('loading dfs')
    exp_df = pd.read_pickle(exp_path)
    print('loaded exp, loading rans')
    rans_df = pd.read_pickle(rans_path)
    print('loaded dfs')
    exp_coords  = _exp_centerline_c123(exp_df, n=n, tol_y=tol_y, x_col=x_col, y_col=y_col)
    print('got exp coords')
    rans_coords = _rans_centerline_c123(rans_df, n=n, tol_y=tol_y, x_col=x_col, y_col=y_col)
    print('got rans coords')
    pred_coords = bcoords_from_pred(y_pred) if y_pred is not None else None

    x_exp,  y_exp  = _xy_from_c123(exp_coords["c1"],  exp_coords["c2"],  exp_coords["c3"])
    x_rans, y_rans = _xy_from_c123(rans_coords["c1"], rans_coords["c2"], rans_coords["c3"])
    print('got xy from rans and exp')
    if pred_coords is not None:
        x_pred, y_pred_xy = _xy_from_c123(pred_coords["c1"], pred_coords["c2"], pred_coords["c3"])

    V1, V2, V3 = _triangle_vertices()
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.plot([V1[0], V2[0], V3[0], V1[0]],
            [V1[1], V2[1], V3[1], V1[1]],
            color="black", lw=1.6)

    # optional faint grid
    for t in np.linspace(0.1, 0.9, 9):
        ax.plot([t*V1[0] + (1-t)*V2[0], t*V1[0] + (1-t)*V3[0]],
                [t*V1[1] + (1-t)*V2[1], t*V1[1] + (1-t)*V3[1]],
                lw=0.5, color="0.9", zorder=1)
        ax.plot([t*V2[0] + (1-t)*V3[0], t*V2[0] + (1-t)*V1[0]],
                [t*V2[1] + (1-t)*V3[1], t*V2[1] + (1-t)*V1[1]],
                lw=0.5, color="0.9", zorder=1)

    ax.scatter(x_rans, y_rans, s=28, marker="^", color="C1", label="RANS", alpha=0.9, zorder=2)
    ax.scatter(x_exp,  y_exp,  s=40, marker="o", facecolors="white", edgecolors="C0",
               linewidths=1.5, label="EXP", zorder=4)
    if pred_coords is not None:
        ax.scatter(x_pred, y_pred_xy, s=32, marker="s", color="C2", label="PRED", alpha=0.9, zorder=3)

    ax.set_aspect("equal", "box")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, np.sqrt(3)/2 + 0.05)
    ax.set_xlabel("Barycentric x")
    ax.set_ylabel("Barycentric y")
    ax.set_title(title or f"Lumley/Barycentric Triangle (centerline, n={n})")
    ax.text(-0.04, -0.02, "2C", fontsize=10)
    ax.text(1.02, -0.02, "1C", fontsize=10)
    ax.text(0.51, np.sqrt(3)/2 + 0.02, "3C", fontsize=10)
    ax.legend(frameon=True)
    fig.tight_layout()

    if save_dir:
        _ensure_dir(save_dir)
        out = os.path.join(save_dir, "lumley_triangle.png")
        fig.savefig(out, dpi=300); print(f"[INFO] saved {out}")
    else:
        plt.show()
    plt.close(fig)
# Plotting/plot_lumley.py

def _triangle_vertices():
    # 1C, 2C, 3C vertices
    return (1.0, 0.0), (0.0, 0.0), (0.5, math.sqrt(3)/2.0)

def _draw_triangle(ax):
    (x1,y1), (x2,y2), (x3,y3) = _triangle_vertices()
    # Draw in solid black, behind data
    ax.plot([x1,x2,x3,x1], [y1,y2,y3,y1], color='k', linewidth=1.5, zorder=1)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, math.sqrt(3)/2.0 + 0.05)
    ax.set_xlabel('Barycentric x')
    ax.set_ylabel('Barycentric y')
    ax.set_title('Lumley / Barycentric Map')


def _to_np(t):
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return t

def _centerline_band_indices(Cx, Cy, tol_y=1.5e-3):
    """Return indices near centerline |Cy| <= tol_y."""
    import numpy as np
    idx = np.where(np.abs(Cy) <= tol_y)[0]
    return idx

def _even_subsample_indices_by_x(Cx, idx, n):
    """Pick ~n evenly spaced points in x among indices idx."""
    import numpy as np
    if len(idx) <= n:
        return idx
    x = Cx[idx]
    order = np.argsort(x)
    idx_sorted = idx[order]
    # pick n evenly spaced indices
    sel = np.linspace(0, len(idx_sorted)-1, num=n, dtype=int)
    return idx_sorted[sel]

def _extract_xy_from_lumley_entry(entry):
    """
    Accepts any of:
      - {'xy': array, 'C': array}
      - (C, xy)
      - {'c1':..., 'c2':..., 'c3':...}  # compute C,xy on the fly
    Returns (C, xy) or (None, None).
    """
    if entry is None:
        return None, None

    # Preferred: already packaged
    if isinstance(entry, dict):
        C = entry.get('C', None)
        xy = entry.get('xy', None)
        if (C is not None) or (xy is not None):
            return C, xy

        # Fallback: c1,c2,c3 → compute C and xy
        if all(k in entry for k in ('c1', 'c2', 'c3')):
            c1 = _to_np(entry['c1'])
            c2 = _to_np(entry['c2'])
            c3 = _to_np(entry['c3'])
            C = np.stack([c1, c2, c3], axis=-1)

            (x1, y1), (x2, y2), (x3, y3) = _triangle_vertices()
            x = c1 * x1 + c2 * x2 + c3 * x3
            y = c1 * y1 + c2 * y2 + c3 * y3
            xy = np.stack([x, y], axis=-1)
            return C, xy

    # Tuple form (C, xy)
    if isinstance(entry, (tuple, list)) and len(entry) == 2:
        C, xy = entry
        return C, xy

    return None, None


def plot_lumley_case(lumley_dict, bary_preds_by_case, grid_dict, best_fin,
                     split: str, case: str,
                     tol_y: float = 1.5e-3, n_pred: int = 200,
                     save_dir: str | None = None,
                     colors: dict | None = None):
    """
    Make ONE triangle plot for a single case, showing RANS, EXP, and PRED.
    Saves to save_dir/lumley_<split>_<case>_<best_fin>.png if save_dir is provided.
    """
    if colors is None:
        colors = {'RANS': 'C0', 'EXP': 'C1', 'PRED': 'C3'}

    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_triangle(ax)  # draws the triangle outline

    # --- RANS & EXP centerline curves from lumley_dict ---
    for src in ('RANS', 'EXP'):
        entry = lumley_dict.get(split, {}).get(src, {}).get(case, None)
        if entry is None:
            continue
        C, xy = _extract_xy_from_lumley_entry(entry)
        xy = _to_np(xy)
        if xy is None or len(xy) == 0:
            continue

        # optional thin path (kept subtle so it doesn't look like the triangle border)
        ax.plot(xy[:, 0], xy[:, 1],
                lw=1.0, alpha=0.9, color=colors.get(src, None), zorder=3)

        # markers so data can't be confused with the triangle edges
        ax.scatter(
            xy[:, 0], xy[:, 1],
            s=18,
            facecolors='white' if src == 'EXP' else colors.get(src, None),
            edgecolors=colors.get(src, None),
            linewidths=1.1,
            label=src,
            zorder=4 if src == 'EXP' else 3
        )

   # --- Predicted points: trim to band and subsample using grid_dict ---
    pred_entry = bary_preds_by_case.get(case, None)
    if pred_entry is not None:
        C_t  = pred_entry.get('C', None)    # [N,1,3] or [N,3] (torch or np) or None
        xy_t = pred_entry.get('xy', None)   # [N,1,2] or [N,2] (torch or np) or None
    
        # to numpy
        C  = _to_np(C_t)   if C_t  is not None else None
        xy = _to_np(xy_t)  if xy_t is not None else None
    
        # squeeze the extra dim added by `maybe(...)`
        if xy is not None and xy.ndim == 3 and xy.shape[1] == 1:
            xy = np.squeeze(xy, axis=1)
        if C is not None and C.ndim == 3 and C.shape[1] == 1:
            C = np.squeeze(C, axis=1)
    
        # if xy not provided, compute from C
        if xy is None and C is not None:
            (x1, y1), (x2, y2), (x3, y3) = _triangle_vertices()
            x = C[:, 0] * x1 + C[:, 1] * x2 + C[:, 2] * x3
            y = C[:, 0] * y1 + C[:, 1] * y2 + C[:, 2] * y3
            xy = np.stack([x, y], axis=-1)
    
        if xy is not None:
            Cx, Cy = grid_dict.get(split, {}).get(case, (None, None))
            if Cx is not None and Cy is not None:
                Cx = np.asarray(Cx); Cy = np.asarray(Cy)
                idx_band = _centerline_band_indices(Cx, Cy, tol_y=tol_y)
                if idx_band.size > 0:
                    idx_sub = _even_subsample_indices_by_x(Cx, idx_band, n=n_pred)
                    pts = xy[idx_sub, :]
                    ax.scatter(pts[:, 0], pts[:, 1],
                               s=12, alpha=0.7, label='PRED',
                               color=colors.get('PRED', None), zorder=5)


    # legend (dedupe)
    handles, labels = ax.get_legend_handles_labels()
    seen, H, L = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); H.append(h); L.append(l)
    ax.legend(H, L, loc='upper right', fontsize=9, frameon=True)

    ax.set_title(f'{split} - {case} - {best_fin} model - Lumley Triangle')
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        safe_best = str(best_fin).replace(' ', '_')
        fname = f"lumley_{split}_{case}_{safe_best}.png"
        fpath = os.path.join(save_dir, fname)
        fig.savefig(fpath, dpi=300)
        print(f"[INFO] saved {fpath}")
        plt.close(fig)
    else:
        plt.show()

    return 


#test
'''
from Trials import TRIALS
trial = TRIALS['single_inter_c2_f2']
idx = '1'
num = '1'
tlist = [f'Case{idx}_FOV{num}']
dnd = 'nondim'
data_dir = os.path.join(PROJECT_ROOT, 'Data', 'Shear_mixing')
rans_dir = os.path.join(data_dir, 'RANS', 'training', 'MLS')
exp_dir = os.path.join(data_dir, 'EXP', 'train_exp',dnd)

for title in tlist:
    case_name, fov = title.split('_')[0], title.split('_')[1]
    rans_fname = f'{case_name}_{dnd}_{fov}.pkl'
    rans_fpath = os.path.join(rans_dir, rans_fname)
    exp_fname = f'{dnd}_{title}.pkl'
    exp_fpath = os.path.join(exp_dir, exp_fname)
    plot_lumley(
        rans_path=rans_fpath,
        exp_path=exp_fpath,
        y_pred=None,              # or {"c1":..., "c2":..., "c3":...}
        save_dir=None,
        title= title,
        n=20,
        tol_y=2.5e-3,
        x_col="Cx",
        y_col="Cy"
    )
'''
