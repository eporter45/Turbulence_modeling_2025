# clean_exp_data.py
"""
Experimental Data Cleaning Pipeline
-----------------------------------
This script:
  - Traverses raw experiment folders to find FOV data
  - Loads and merges Mean + Higher data for each FOV
  - Converts coordinates (x_mm, y_mm → Cx, Cy [m])
  - Saves combined FOVs and an overall case-wide dataframe

Author: Elliot Porter
"""

import os
import sys
import re
import pandas as pd
import matplotlib.pyplot as plt

# --- Resolve project root ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# --- Directory setup ---
DATA_ROOT = os.path.join(PROJECT_ROOT, "Data", "Shear_mixing")
RAW_EXP_DIR = os.path.join(DATA_ROOT, "Raw_data", "exp")
CLEAN_EXP_DIR = os.path.join(DATA_ROOT, "EXP_ex")
os.makedirs(CLEAN_EXP_DIR, exist_ok=True)


# ==========================================================
# Helper functions
# ==========================================================
def get_nested_dir(parent_dir, startswith):
    """Find the first subdirectory starting with a specific prefix."""
    for entry in os.listdir(parent_dir):
        full_path = os.path.join(parent_dir, entry)
        if os.path.isdir(full_path) and entry.startswith(startswith) and not entry.startswith("._"):
            return full_path
    raise FileNotFoundError(f"No valid subdirectory found in {parent_dir} starting with '{startswith}'")


def get_first_data_line_idx(file_path, encoding='cp1252'):
    """Find first line containing numeric data."""
    with open(file_path, 'r', encoding=encoding) as f:
        for i, line in enumerate(f):
            if re.match(r'^\s*[-+]?[0-9]', line):
                return i
    raise ValueError(f"No numeric data found in {file_path}")


def load_fovs(data_dir, mode='Mean'):
    """Load all FOVs for a given mode ('Mean' or 'Higher')."""
    fov_data = {}
    if mode == 'Mean':
        column_names = ["x_mm", "y_mm", "U", "V", "W", "V_mag", "uu", "vv", "ww", "uv"]
    elif mode == 'Higher':
        column_names = ["x_mm", "y_mm", "uuu", "uvv", "uww", "vuu", "vvv", "vww", "uuuu", "vvvv", "wwww"]
    else:
        raise ValueError(f"Invalid mode: {mode}")

    for filename in os.listdir(data_dir):
        if filename.startswith("._") or not filename.startswith("Side-View"):
            continue

        match = re.search(r"FOV\s*(\d+)\s*\[x\s*=\s*(\d+)-(\d+)\s*mm\]", filename)
        if not match:
            continue

        fov_num = int(match.group(1))
        file_path = os.path.join(data_dir, filename)
        first_row = get_first_data_line_idx(file_path)
        df = pd.read_csv(file_path, sep=r"\s+", skiprows=first_row, names=column_names, engine="python", encoding="cp1252")
        fov_data[fov_num] = df
    return fov_data


def convert_coords(df):
    """Convert mm → meters and rename columns."""
    df["Cx"] = df["x_mm"] / 1000.0
    df["Cy"] = df["y_mm"] / 1000.0
    df.drop(columns=["x_mm", "y_mm"], inplace=True)
    return df


def merge_mean_and_higher(mean_fovs, higher_fovs):
    """Merge Mean and Higher datasets per FOV on (x_mm, y_mm)."""
    merged_fovs = {}
    for fov_num, mean_df in mean_fovs.items():
        high_df = higher_fovs.get(fov_num)
        if high_df is None:
            print(f"[WARN] No Higher data for FOV{fov_num}")
            merged = mean_df
        else:
            merged = pd.merge(mean_df, high_df, on=["x_mm", "y_mm"], how="outer")
        merged_fovs[fov_num] = convert_coords(merged)
    return merged_fovs


def combine_fovs(fov_data):
    """Combine all merged FOVs into one DataFrame."""
    combined_df = pd.concat(fov_data.values(), ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["Cx", "Cy"])
    return combined_df.sort_values(by=["Cx", "Cy"]).reset_index(drop=True)


def save_fovs_as_pickle(fov_data, case_name, save_dir):
    """Save merged FOVs and overall combined DataFrame."""
    os.makedirs(save_dir, exist_ok=True)
    combined_df = pd.DataFrame()

    for fov, df in fov_data.items():
        filename = f"{case_name}_FOV{fov}.pkl"
        filepath = os.path.join(save_dir, filename)
        df.to_pickle(filepath)
        print(f"[SAVED] {filepath}")
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    combined_path = os.path.join(save_dir, f"{case_name}_combined.pkl")
    print(f"[Info] Cx range: ({combined_df['Cx'].min():.4f}, {combined_df['Cx'].max():.4f})")
    print(f"[Info] Cy range: ({combined_df['Cy'].min():.4f}, {combined_df['Cy'].max():.4f})")
    combined_df.to_pickle(combined_path)
    print(f"[SAVED] Combined dataframe: {combined_path}")


# ==========================================================
# Main cleaning function
# ==========================================================
def clean_exp_case(case, save_dir=CLEAN_EXP_DIR, plot=False):
    """Load, clean, merge, and save experiment data for one case."""
    print(f"\n[START] Cleaning experimental data for {case}")

    case_dir = os.path.join(RAW_EXP_DIR, case)
    nested_case_dir = get_nested_dir(case_dir, "Case")
    mean_dir = get_nested_dir(get_nested_dir(nested_case_dir, "Mean"), "Mean Velocity")
    higher_dir = get_nested_dir(get_nested_dir(nested_case_dir, "Mean"), "Higher")

    mean_fovs = load_fovs(mean_dir, "Mean")
    higher_fovs = load_fovs(higher_dir, "Higher")

    merged_fovs = merge_mean_and_higher(mean_fovs, higher_fovs)
    save_fovs_as_pickle(merged_fovs, case, save_dir)

    if plot:
        combined_df = pd.concat(merged_fovs.values(), ignore_index=True)
        plt.figure(figsize=(6, 5))
        plt.scatter(combined_df["Cx"], combined_df["Cy"], s=2, c=combined_df.get("U", 0))
        plt.title(f"{case} - Velocity Field (merged)")
        plt.xlabel("Cx [m]")
        plt.ylabel("Cy [m]")
        plt.tight_layout()
        plt.show()

    print(f"[DONE] ✅ Cleaned and merged experimental data for {case}")


# ==========================================================
# Run
# ==========================================================
if __name__ == "__main__":
    cases = ["Case1", "Case2"]
    for case in cases:
        clean_exp_case(case, plot=False)
