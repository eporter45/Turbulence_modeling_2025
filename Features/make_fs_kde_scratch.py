# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 14:21:10 2025

@author: eoporter
"""

import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import pickle as pkl
from Features.make_featuresets import get_feature_set
#Dynamically resolve project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add to system path if not already
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    

def simple_kde_plot(df, config, grad_type='', d_nd='', save_dir='', case_name="", save=False):
    """
    Overlays KDE curves for all features from config on a single plot.

    Args:
        df (pd.DataFrame): Input DataFrame with raw features.
        config (dict): Feature config with keys 'input' and optionally 'trim_z'.
        save_dir (str or Path): Directory to save the plot.
        case_name (str): Optional name to label the plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    feature_list = get_feature_set(config)
    print(f'Feat list: \n {feature_list}')
    fs = config['features']['input']
    plt.figure(figsize=(10, 6))

    for feature in feature_list:
        if feature not in df.columns:
            print(f"[Warning] Feature '{feature}' not found in DataFrame. Skipping.")
            continue
        
        plt.figure(figsize=(6, 4))  # <-- New figure per feature
        sns.kdeplot(df[feature].dropna(), fill=True, linewidth=2)
        plt.title(f"KDE:{grad_type} {d_nd} {fs} {feature}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        
        plot_name = f"kde_{feature}_{case_name}.png"
        if save:
            if not save_dir:
                raise ValueError('[Error] No name was passed into save_dir')
            plt.savefig(os.path.join(save_dir, plot_name))
        else:
            plt.show()
    
    plt.close()


def staggered_kde_plot(df1, df2, config,bds=(2.5, 97.5), grad_labels=('MLS', 'OG'), d_nd='', save_dir='', case_name='', save=False):
    """
    Alternates KDE plots between two DataFrames for each feature in config.

    Args:
        df1, df2 (pd.DataFrame): Input DataFrames (e.g., MLS and OG).
        config (dict): Feature config (must contain "features": {"input": ...}).
        grad_labels (tuple): Labels for df1 and df2 (e.g., ('MLS', 'OG')).
        d_nd (str): Dim/nondim flag.
        save_dir (str): Directory to sa,ve the plots.
        case_name (str): Case name to include in title and filename.
        save (bool): Whether to save the plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    pct_bounds = [bds[0], bds[1]]
    feature_list = get_feature_set(config)
    fs = config['features']['input']

    for feature in feature_list:
        if feature not in df1.columns or feature not in df2.columns:
            print(f"[Warning] Feature '{feature}' missing in one or both DataFrames. Skipping.")
            continue
        
        # Percentile clipping (2.5% - 97.5%)
        vals1 = df1[feature].dropna()
        vals2 = df2[feature].dropna()

        lower1, upper1 = np.percentile(vals1, pct_bounds)
        lower2, upper2 = np.percentile(vals2, pct_bounds)
        
        trimmed1 = vals1[(vals1 >= lower1) & (vals1 <= upper1)]
        trimmed2 = vals2[(vals2 >= lower2) & (vals2 <= upper2)]

        plt.figure(figsize=(8, 5))
        sns.kdeplot(trimmed1, label=grad_labels[0], linewidth=2)
        sns.kdeplot(trimmed2, label=grad_labels[1], linewidth=2)

        plt.title(f"KDE of {feature} | {grad_labels[0]} vs {grad_labels[1]} | {d_nd}, {fs}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        
        plot_name = f'kde_{feature}_{case_name}_{d_nd}_{fs}.png'
        if save:
            if not save_dir:
                raise ValueError('[Error] No name was passed into save_dir')
            plt.savefig(os.path.join(save_dir, plot_name))
        else:
            plt.show()

        plt.close()

def multi_feature_kde_plot(df, config, grad_type='', d_nd='', save_dir='', case_name="",
                           bounds=(0.0, 100.0), save=False):
    """
    Plots KDE curves for all selected features on a single shared plot.
    Optionally applies percentile clipping to trim outliers.

    Args:
        df (pd.DataFrame): DataFrame containing raw features.
        config (dict): Feature config, must contain "features": {"input": ...}.
        grad_type (str): Label for gradient type (e.g., 'MLS' or 'OG').
        d_nd (str): 'dim' or 'nondim'.
        save_dir (str): Directory to save the figure.
        case_name (str): Optional case identifier.
        bounds (tuple): Percentile bounds (lower, upper) for trimming. Defaults to (0, 100).
        save (bool): Whether to save the plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    feature_list = get_feature_set(config)
    fs = config['features']['input']
    lower_pct, upper_pct = bounds

    plt.figure(figsize=(10, 6))
    for feature in feature_list:
        if feature not in df.columns:
            print(f"[Warning] Feature '{feature}' not found in DataFrame. Skipping.")
            continue

        vals = df[feature].dropna()
        
        if bounds != (0.0, 100.0):  # Apply clipping only if needed
            lower, upper = np.percentile(vals, [lower_pct, upper_pct])
            vals = vals[(vals >= lower) & (vals <= upper)]

        sns.kdeplot(vals, label=feature, linewidth=2)

    plt.title(f"Overlay KDE | {grad_type}, {d_nd}, {fs} | {case_name}")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()

    plot_name = f'overlay_kde_{grad_type}_{case_name}_{d_nd}_{fs}.png'
    if save:
        if not save_dir:
            raise ValueError('[Error] No name was passed into save_dir')
        plt.savefig(os.path.join(save_dir, plot_name))
        print(f"[SAVED] KDE plot saved at {os.path.join(save_dir, plot_name)}")
    else:
        plt.show()

    plt.close()


#Plotting config and setup

sets =['FS1', 'FS1_simple', 'FS2', 'FS3a',
       'FS3b', 'FS3c', 'FS4', 'FS5',
       'FS6', 'FS7']
sets = ['FS8']
#config for running
idx = '1'
case = f'Case{idx}'
grad_type = 'MLS' #mls or og 
d_nd = 'nondim' #dim or nondim
fov='FOV1'
filename = f'{case}_{d_nd}_{fov}.pkl'
save=True


#load RANS data
rans_dir = os.path.join(PROJECT_ROOT, 'data', 'Shear_mixing', 'RANS', 'training')
mls_file_dir = os.path.join(rans_dir, 'MLS')
og_file_dir = os.path.join(rans_dir, 'og')
mls_filepath = os.path.join(mls_file_dir, filename)
og_filepath = os.path.join(og_file_dir, filename)
mls_df = pd.read_pickle(mls_filepath)
og_df = pd.read_pickle(og_filepath)
save_dir = os.path.join(rans_dir, 'kde')
#config setup and plotting
save = False
bds = (0.0, 100.0)
sets = ['FS8']
feature_sets = {}
'''
feature_sets = {}
for fs in sets:
    config = {
        "features": {
            "input": fs,      # or "FS3b", etc.
            "trim_z": True       # Optional z-trimming
        }
    }
    feature_sets[fs] = get_feature_set(config)
    multi_feature_kde_plot(og_df, config=config, grad_type='OG',
                       d_nd=d_nd, save_dir=save_dir,
                       case_name='Case1', bounds=bds, save=save)
    multi_feature_kde_plot(mls_df, config=config, grad_type='MLS',
                       d_nd=d_nd, save_dir=save_dir,
                       case_name='Case1', bounds=bds, save=save)
'''
def find_tail_percentile_asymmetric(data,skew_threshold=1.0, skew=None, step=1, kurt_threshold=1, max_trim=50):
    """
    Incrementally trim data from one tail based on skew direction
    to find the percentile where kurtosis falls below a threshold.

    Args:
        data (pd.Series or np.array): Input 1D data array.
        skew (float): Precomputed skewness of the data. If None, it will be calculated.
        step (float): Percent to trim at each iteration (from one end).
        kurt_threshold (float): Threshold for acceptable excess kurtosis.
        max_trim (float): Max percent to trim.

    Returns:
        trim_percent (float): Final trim percent from one tail.
        kurt_vals (list): List of kurtosis values per step.
        trim_percents (list): List of tried trim percents.
        direction (str): Which side was trimmed ('left' or 'right').
    """
    from scipy.stats import kurtosis
    data_sorted = np.sort(data.dropna())
    n = len(data_sorted)
    
    if skew is None:
        skew = data.skew()
        
    # Determine which tail is heavier
    direction = 'right' if skew > 0 else 'left'

    kurt_vals = []
    trim_percents = []

    trim_percent = 0
    while trim_percent <= max_trim:
        if direction == 'right':
            # Trim from upper tail
            upper_idx = int(n * (1 - trim_percent / 100))
            trimmed_data = data_sorted[:upper_idx]
        else:
            # Trim from lower tail
            lower_idx = int(n * trim_percent / 100)
            trimmed_data = data_sorted[lower_idx:]

        if len(trimmed_data) < 10:
            break

        k = kurtosis(trimmed_data, fisher=True)
        kurt_vals.append(k)
        trim_percents.append(trim_percent)

        if abs(k) > kurt_threshold or abs(skew) > skew_threshold:
            return trim_percent, kurt_vals, trim_percents, direction

        trim_percent += step

    return trim_percent, kurt_vals, trim_percents, direction


from scipy.stats import kurtosis, skew

def analyze_feature_trims(df, feature_sets, skew_threshold, kurt_threshold=1.5,
                          compute_healing_profile=False, healing_args=None,
                          save_plots=False, plot_dir=""):
    results = {}

    for fs_name, feature_list in feature_sets.items():
        print(f"\n[INFO] Analyzing Feature Set: {fs_name}")
        fs_results = {}

        for feature in feature_list:
            if feature not in df.columns:
                print(f"[WARNING] Feature '{feature}' not in DataFrame. Skipping.")
                continue

            vals = df[feature].dropna()
            s = skew(vals)
            k = kurtosis(vals, fisher=True)

            trim_pct, kurt_vals, trim_percents, direction = find_tail_percentile_asymmetric(
                vals, skew=s, skew_threshold=skew_threshold, kurt_threshold=kurt_threshold
            )

            sorted_vals = np.sort(vals)
            n = len(sorted_vals)

            if direction == 'right':
                upper_idx = int(n * (1 - trim_pct / 100))
                trimmed_data = sorted_vals[:upper_idx]
            else:
                lower_idx = int(n * trim_pct / 100)
                trimmed_data = sorted_vals[lower_idx:]

            min_val, max_val = trimmed_data.min(), trimmed_data.max()

            fs_results[feature] = {
                'original_skew': s,
                'original_kurtosis': k,
                'trim_percent': trim_pct,
                'trim_direction': direction,
                'trimmed_kurtosis': kurt_vals[-1] if kurt_vals else None,
                'min_bulk': min_val,
                'max_bulk': max_val
            }

            print(f"  [OK] {feature}: skew={s:.2f}, kurt={k:.2f}, trimmed {direction} {trim_pct}%, bulk=({min_val:.3g}, {max_val:.3g})")

            # Compute healing profile
            if compute_healing_profile:
                h_args = healing_args if healing_args else {}
                hp = healing_profile(vals, direction=direction, feature_name=feature,
                                      **h_args)
                fs_results[feature]['healing_profile'] = hp

                # Extract and sort
                percentiles = list(hp.keys())
                healing_lengths = list(hp.values())

                sorted_pairs = sorted(
                    zip(percentiles, healing_lengths),
                    key=lambda x: float(str(x[0]).replace('%', ''))
                )
                percentiles, healing_lengths = zip(*sorted_pairs)

                # Plot healing profile
                plt.plot(percentiles, healing_lengths, marker='o', label='Healing Length')
                plt.title(f"Healing Profile for {feature}")

                # Label formatting
                if isinstance(percentiles[0], str) and percentiles[0].endswith("%"):
                    plt.xticks(rotation=45)
                    plt.xlabel("Start Percentile from Mode")
                else:
                    plt.xlabel("Start Percentile")

                plt.ylabel("Healing Length Percentile")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()

                if save_plots and plot_dir:
                    os.makedirs(plot_dir, exist_ok=True)
                    fname = os.path.join(plot_dir, f"{feature}_healing_profile.png")
                    plt.savefig(fname)
                    plt.close()
                else:
                    plt.show()

            # Plot KDE with trimmed region
            if save_plots:
                sns.kdeplot(vals, label='Raw', linewidth=2)
                sns.kdeplot(trimmed_data, label='Trimmed', linestyle='--', linewidth=2)
                plt.axvline(min_val, color='gray', linestyle='--')
                plt.axvline(max_val, color='gray', linestyle='--')
                plt.title(f"KDE: {feature} (Trimmed {direction} {trim_pct}%)")
                plt.xlabel("Value")
                plt.ylabel("Density")
                plt.legend()
                plt.grid(True)
                if plot_dir:
                    os.makedirs(plot_dir, exist_ok=True)
                    fname = os.path.join(plot_dir, f"{feature}_trimmed_kde.png")
                    plt.savefig(fname)
                else:
                    plt.show()
                plt.close()

                # Plot kurtosis evolution
                plt.plot(trim_percents, kurt_vals, marker='o')
                plt.axhline(y=kurt_threshold, color='r', linestyle='--', label='Threshold')
                plt.title(f"Kurtosis vs Trim: {feature}")
                plt.xlabel(f"Trimmed % from {direction}")
                plt.ylabel("Excess Kurtosis")
                plt.grid(True)
                plt.legend()
                if plot_dir:
                    fname = os.path.join(plot_dir, f"{feature}_kurtosis_vs_trim.png")
                    plt.savefig(fname)
                else:
                    plt.show()
                plt.close()

        results[fs_name] = fs_results

    return results



from scipy.stats import gaussian_kde, skew as calc_skew, percentileofscore

def healing_profile(data, direction='right', feature_name='Unknown',
                    decay_threshold=0.33, max_steps=20, fine_step=1):
    """
    Computes healing profile starting from the mode percentile, stepping toward the tail,
    and calculating healing length via KDE density decay.

    Args:
        data (array-like): 1D numeric data.
        direction (str): 'right' or 'left' — which tail to analyze.
        feature_name (str): Feature name for logging.
        decay_threshold (float): Fraction of density drop used to define healing.
        max_steps (int): Max number of percentile steps to move from the mode.
        fine_step (int): Step size in percentiles (1 = every percentile).

    Returns:
        dict: {start_percentile: healing_percentile}
    """
    from scipy.stats import percentileofscore

    data = np.sort(np.array(data))
    n = len(data)
    results = {}

    kde = gaussian_kde(data)
    x_grid = np.linspace(data.min(), data.max(), 1000)
    kde_vals = kde(x_grid)

    # Get mode via KDE max
    mode_idx = np.argmax(kde_vals)
    mode_val = x_grid[mode_idx]
    mode_percentile = percentileofscore(data, mode_val, kind='mean')
    skew_val = calc_skew(data)

    print(f"[INFO] Feature: {feature_name}")
    print(f"       Mode value: {mode_val:.4f}")
    print(f"       Mode percentile: {mode_percentile:.2f}%")
    print(f"       Skew: {skew_val:.3f}")

    # Step toward the skewed tail
    if direction == 'right':
        start_range = np.arange(mode_percentile, min(100, mode_percentile + max_steps * fine_step + 1), fine_step)
    else:
        start_range = np.arange(mode_percentile, max(0, mode_percentile - max_steps * fine_step - 1), -fine_step)

    for p in start_range:
        start_val = np.percentile(data, p)

        if direction == 'right':
            search_grid = x_grid[x_grid >= start_val]
            search_vals = kde(search_grid)
        else:
            search_grid = x_grid[x_grid <= start_val][::-1]
            search_vals = kde(search_grid)

        if len(search_vals) == 0:
            results[f"{p:.1f}%"] = None
            continue

        start_density = kde(start_val)[0]
        threshold_density = decay_threshold * start_density

        heal_idx = np.argmax(search_vals <= threshold_density)
        if heal_idx == 0 and search_vals[0] > threshold_density:
            healing_p = 100
        else:
            heal_val = search_grid[heal_idx]
            healing_p = 100 * (np.searchsorted(data, heal_val) / (n - 1))

        results[f"{p:.1f}%"] = healing_p

    return results




# Then normalize all features with these global bounds:
def normalize_with_trimmed_bulk(df, feature_sets, trim_results, save_dir="", tag="TrimmedBulk_Normalized", show=False):
    import os
    import seaborn as sns
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    fs_bounds = {}
    for fs_name, feats in feature_sets.items():
        fs_bounds[fs_name] = {}
        bulk_info = fs_bounds[fs_name]
        min_bulk = []
        max_bulk = []
        plt.figure(figsize=(10, 6))
        print(f"\n[PLOTTING] {fs_name} normalized KDEs with trimmed bulk bounds")

        for feature in feats:
            if feature not in df.columns:
                print(f"[SKIP] {feature} missing in data")
                continue

            # Get trimmed bulk min and max for this feature
            if fs_name not in trim_results or feature not in trim_results[fs_name]:
                print(f"[SKIP] No trim results for {feature} in {fs_name}")
                continue

            feat_min_bulk = trim_results[fs_name][feature]['min_bulk']
            feat_max_bulk = trim_results[fs_name][feature]['max_bulk']
            min_bulk.append(feat_min_bulk)
            max_bulk.append(feat_max_bulk)
            if feat_min_bulk == feat_max_bulk:
                print(f"[WARN] Zero range for trimmed bulk normalization. Skipping {feature}.")
                continue

            data = df[feature].dropna()

            # Normalize using trimmed bulk min/max
            normed = (data - feat_min_bulk) / (feat_max_bulk - feat_min_bulk)
            normed = normed.clip(0, 1)  # Clip to [0,1]

            sns.kdeplot(normed, label=feature, linewidth=2)
        
        bulk_info['max'] = max(max_bulk)
        bulk_info['min'] = min(min_bulk)
        plt.title(f"{tag} KDE | {fs_name}")
        plt.xlabel("Normalized Value")
        plt.ylabel("Density")
        plt.legend(fontsize='small')
        plt.grid(True)
        plt.tight_layout()

        fname = os.path.join(save_dir, f"{tag}_{fs_name}_kde.png")
        if show:
            plt.show()
        else:
            plt.savefig(fname)
            print(f"[SAVED] {fname}")
        plt.close()
        
    return fs_bounds
# Call this function with global bounds


#kurt_thresh = 0.7
#skew_thresh = 0.2
'''
trim_results = analyze_feature_trims(
    mls_df,
    feature_sets,
    skew_threshold=skew_thresh,
    kurt_threshold=kurt_thresh,
    compute_healing_profile=True,
    healing_args={
        'decay_threshold': 0.33,
        'max_steps': 10,       # optional
        'fine_step': 2         # percent step size (like the old 'step')
    },
    save_plots=False,
    plot_dir=os.path.join(rans_dir, 'kde_trimmed')
)'''


#norm_plot_dir = os.path.join(rans_dir, 'kde_normalized')
#normalize_and_plot_trimmed_features(mls_df, trim_results, feature_sets,
#                                   save_dir=norm_plot_dir, tag='MLS_Normalized', show=True)
#fs_bounds = normalize_with_trimmed_bulk(mls_df, feature_sets, trim_results,
#                           save_dir=norm_plot_dir, tag='MLS_TrimmedBulkNormalized', show=True)
def trim_plot_global_bulk(df, fs_bounds, feature_sets, show=True, save_dir=None, tag=None):
    """
    Plots KDEs of features normalized by their global bulk bounds (from healing profile).

    Args:
        df (pd.DataFrame): Input DataFrame.
        fs_bounds (dict): Output from normalize_with_trimmed_bulk() with global min/max.
        feature_sets (dict): Feature sets (dict of feature lists).
        show (bool): Whether to display plots inline.
        save_dir (str): Optional directory to save plots.
        tag (str): Optional tag for filenames.
    """
    for fs_name, bounds in fs_bounds.items():
        feats = feature_sets.get(fs_name, [])
        bulk_min = bounds['min']
        bulk_max = bounds['max']
        denom = bulk_max - bulk_min
        if denom == 0:
            print(f"[WARNING] Skipping {fs_name} — zero range in bulk bounds.")
            continue

        print(f'\n[INFO] {fs_name} Global Bulk Bounds: ({bulk_min:.3f}, {bulk_max:.3f})')

        # Normalize all features in this feature set to [0, 1] based on bulk bounds
        norm_feats = {}
        for feat in feats:
            if feat not in df.columns:
                print(f"[WARNING] {feat} missing from DataFrame. Skipping.")
                continue
            norm_feats[feat] = (df[feat] - bulk_min) / denom

        # Plot
        plt.figure(figsize=(10, 6))
        print(f"[PLOTTING] {fs_name}: Normalized KDEs using global trimmed bulk bounds")

        for feat, norm_vals in norm_feats.items():
            sns.kdeplot(norm_vals.dropna(), label=feat, linewidth=2)

        plt.axvline(0, color='gray', linestyle='--', linewidth=1)
        plt.axvline(1, color='gray', linestyle='--', linewidth=1)

        plt.title(f'{fs_name} — Normalized KDEs (Trimmed Bulk ∈ [0, 1])')
        plt.xlabel("Normalized Feature Value")
        plt.ylabel("Density")
        plt.legend(fontsize='small')
        plt.grid(True)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fname = os.path.join(save_dir, f"{fs_name}_kde_trimmed_bulk_{tag or ''}.png")
            plt.savefig(fname)
            print(f"[SAVED] Plot saved to: {fname}")
            plt.close()
        elif show:
            plt.show()
  
'''
trim_plot_global_bulk(
    mls_df,
    fs_bounds,
    feature_sets,
    show=True,
    save_dir=os.path.join(rans_dir, 'kde_trimmed_bulk_global'),
    tag='MLS_TrimmedBulkNormalized'
)'''

def process_healing_data(results, smooth=0, print_yn=False):
    from scipy.interpolate import UnivariateSpline
    elbow_points = {}
    for fs in results:
        info = results[fs].keys()
        feats = results[fs]
        for feat in feats:
            info = feats[feat]
            elbow_points[feat] = {}
            elbow = elbow_points[feat]
            hp = info['healing_profile']
        
            # Parse x and y from healing profile
            x_vals = np.array([float(k.replace('%', '')) for k in hp.keys()])
            y_vals = np.array(list(hp.values()))
        
            # Sort to ensure increasing x
            sorted_indices = np.argsort(x_vals)
            x_vals = x_vals[sorted_indices]
            y_vals = y_vals[sorted_indices]
        
            # Interpolate KDE with smoothing
            spline = UnivariateSpline(x_vals, y_vals, s=1e-4)
            smoothed_y = spline(x_vals)
            slopes = spline.derivative(n=1)(x_vals)
        
            # Get mode (peak of smoothed KDE)
            mode_idx = np.argmax(smoothed_y)
            mode_val = x_vals[mode_idx]
            skew = info.get('skew', 0.0)
        
            slope_threshold = 0.01 * np.max(np.abs(slopes))  # Small threshold to detect "flat"
        
            elbow_idx = None
            if skew > 0:
                # Right-tailed → search after the mode
                for i in range(mode_idx + 1, len(slopes)):
                    if abs(slopes[i]) < slope_threshold:
                        elbow_idx = i
                        break
            else:
                # Left-tailed → search *from end back to mode*
                for i in reversed(range(mode_idx)):
                    if abs(slopes[i]) < slope_threshold:
                        elbow_idx = i
                        break
        
            if elbow_idx is None:
                print(f"[WARN] No leveling-off point found for {feat}, using fallback.")
                elbow_idx = mode_idx + 1 if skew > 0 else mode_idx - 1
                elbow_idx = max(0, min(elbow_idx, len(x_vals) - 1))  # clamp
        
            elbow_x = x_vals[elbow_idx]
            elbow_y = smoothed_y[elbow_idx]
        
            # Save elbow location
            elbow['x'] = round(elbow_x, 2)
            if feat=='dp_dx':
                elbow['x'] = round(37.8)
            elbow['y'] = elbow_y
        
            if print_yn:
                tail = "right" if skew > 0 else "left"
                print(f"[INFO] feat {feat} has an elbow at x: {elbow['x']} (percentile), y: {elbow['y']:.2f} on {tail} tail")
                
    return elbow_points
    
#elbow_points = process_healing_data(trim_results, smooth=0.1, print_yn=True)


def plot_kde_with_tail(df, feature, elbow_percentile, tail_direction):
    data = df[feature].dropna()
    
    # Compute elbow value in data scale
    elbow_value = np.percentile(data, elbow_percentile)
    
    # Plot KDE
    sns.kdeplot(data, label='Raw Data')
    
    # Get KDE values for shading
    kde = sns.kdeplot(data).get_lines()[-1]  # last plot line
    x_vals = kde.get_xdata()
    y_vals = kde.get_ydata()
    plt.clf()  # clear the above plot since we just needed the data
    
    # Shade tail region
    if tail_direction == 'right':
        mask = x_vals >= elbow_value
    else:
        mask = x_vals <= elbow_value

    plt.plot(x_vals, y_vals, label='Raw Data')
    plt.fill_between(x_vals[mask], 0, y_vals[mask], color='orange', alpha=0.3, label='Tail Region')
    
    # Mark elbow with a dot on KDE curve
    # Find closest x to elbow_value for plotting the dot
    idx = np.argmin(np.abs(x_vals - elbow_value))
    plt.plot(x_vals[idx], y_vals[idx], 'ro', label='Elbow Point')
    
    plt.xlabel(f'{feature} Value')
    plt.ylabel('Density')
    plt.title(f'KDE & Tail Region for {feature} ({tail_direction} tail, elbow at {elbow_percentile:.1f}%)')
    plt.legend()
    plt.show()
    
def plot_predicted_tails(df, elbow_dict, feature_sets, trim_results):
    print(f'INFO feature set keys: {feature_sets.keys()}')
    for fs in feature_sets:
        print(f'INFO fs : {fs}')
        feats = feature_sets[fs]
        for feat in feats:
            elbow_x = elbow_dict[feat]['x']
            direction = trim_results[fs][feat]['trim_direction']
            plot_kde_with_tail(df, feat, elbow_x, direction)
        
#plot_predicted_tails(mls_df, elbow_points, feature_sets, trim_results)