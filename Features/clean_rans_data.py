# clean_rans.py
"""
RANS Data Cleaning Pipeline
---------------------------
Loads raw CFD output, standardizes column names, shifts, crops to EXP domain,
and saves cleaned .pkl files for downstream feature engineering.

Author: Elliot Porter
"""

import os
import sys
import pandas as pd

# --- Resolve project root ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
print(f"[ROOT] {PROJECT_ROOT}")

# --- Directory setup ---
DATA_ROOT = os.path.join(PROJECT_ROOT, "Data", "Shear_mixing")
RAW_RANS_DIR = os.path.join(DATA_ROOT, "Raw_data", "RANS")
CLEAN_RANS_DIR = os.path.join(DATA_ROOT, "EX_RANS", "cleaned")
os.makedirs(CLEAN_RANS_DIR, exist_ok=True)


# ==========================================================
# Function Definitions
# ==========================================================

def load_raw_rans(case):
    """Load a single raw CFD results text file for a case."""
    filename = f"CFD_{case}_Results.txt"
    file_path = os.path.join(RAW_RANS_DIR, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[ERROR] Missing raw file for {case} at {file_path}")
    print(f"[INFO] Loading {file_path}")
    df = pd.read_csv(file_path, header=0)
    df.columns = df.columns.str.strip()
    return df


def rename_rans_columns(df):
    """Standardize RANS column names."""
    rename_map = {
        'x-coordinate': 'Cx',
        'y-coordinate': 'Cy',
        'z-coordinate': 'Cz',
        'rexx': 'uu',
        'reyy': 'vv',
        'rezz': 'ww',
        'rexy': 'uv',
        'density': 'rho',
        'x-velocity': 'Ux',
        'y-velocity': 'Uy',
        'z-velocity': 'Uz',
        'turb-kinetic-energy': 'k',
        'specific-diss-rate': 'omega',
        'viscosity-turb': 'mu_t',
        'dx-velocity-dx': 'dUx_dx',
        'dy-velocity-dx': 'dUy_dx',
        'dz-velocity-dx': 'dUz_dx',
        'dx-velocity-dy': 'dUx_dy',
        'dy-velocity-dy': 'dUy_dy',
        'dz-velocity-dy': 'dUz_dy',
        'dx-velocity-dz': 'dUx_dz',
        'dy-velocity-dz': 'dUy_dz',
        'dz-velocity-dz': 'dUz_dz',
        'dp-dx': 'dp_dx',
        'dp-dy': 'dp_dy',
        'dp-dz': 'dp_dz',
        'mach': 'Mach',
        'pressure': 'p',
        'temperature': 'T',
        'cellnumber': 'cellID'
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def shift_rans_position(case, df):
    """Shift RANS domain, then crop to match experimental FOV region."""
    rans_shift = {'Case1': 0.3628, 'Case2': 0.3628, 'Case3': 0.0000}
    df['Cx'] = df['Cx'] - rans_shift.get(case, 0.0)

    # Try to load experimental combined bounds
    exp_combined_path = os.path.join(PROJECT_ROOT, "Data", "Shear_mixing", "EXP_ex", f"{case}_combined.pkl")
    if not os.path.exists(exp_combined_path):
        print(f"[WARN] EXP combined file not found for {case}, skipping cropping.")
        return df

    exp_df = pd.read_pickle(exp_combined_path)
    x_min, x_max = exp_df["Cx"].min(), exp_df["Cx"].max()
    y_min, y_max = exp_df["Cy"].min(), exp_df["Cy"].max()

    # Crop RANS to within ±0.1 m margin of EXP bounds
    margin = 0.1
    df = df[
        (df["Cx"] >= x_min - margin) & (df["Cx"] <= x_max + margin) &
        (df["Cy"] >= y_min - margin) & (df["Cy"] <= y_max + margin)
    ]

    print(f"[INFO] Cropped RANS domain to within ±{margin} m of EXP bounds.")
    print(f"[TEST] New Cx range: ({df['Cx'].min():.4f}, {df['Cx'].max():.4f})")
    print(f"[TEST] New Cy range: ({df['Cy'].min():.4f}, {df['Cy'].max():.4f})")
    return df


def clean_rans_case(case, save=True):
    """Load, rename, clean, shift, crop, and save a single RANS case."""
    print(f"\n[START] Cleaning RANS data for {case}")

    df = load_raw_rans(case)
    df = rename_rans_columns(df)

    if "Cx" not in df.columns or "Cy" not in df.columns:
        raise KeyError(f"[ERROR] Missing coordinates in {case}")

    # Basic cleanup
    df = df.dropna(subset=["Ux", "Uy"]).reset_index(drop=True)
    print(f"[INFO] Loaded {len(df)} valid points for {case}")

    # Apply shift and crop
    df = shift_rans_position(case, df)

    print(f"[FINAL] Cx range: ({df['Cx'].min():.4f}, {df['Cx'].max():.4f})")
    print(f"[FINAL] Cy range: ({df['Cy'].min():.4f}, {df['Cy'].max():.4f})")

    # Save cleaned file
    if save:
        save_path = os.path.join(CLEAN_RANS_DIR, f"{case}_pre_grads.pkl")
        df.to_pickle(save_path)
        print(f"[SAVED] {case} → {save_path}")

    print(f"[DONE] ✅ Finished cleaning for {case}")
    return df


def batch_clean_rans(cases):
    """Clean multiple RANS cases sequentially."""
    for case in cases:
        clean_rans_case(case, save=True)


# ==========================================================
# Main Execution
# ==========================================================
if __name__ == "__main__":
    cases = ["Case1", "Case2"]
    batch_clean_rans(cases)
