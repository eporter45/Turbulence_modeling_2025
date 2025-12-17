# ===============================================================
# io_tools.py
# Centralized I/O helpers for RANS–EXP feature engineering pipeline
# ===============================================================

import os
import pandas as pd
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------
# === Path Helpers ==============================================
# ---------------------------------------------------------------
def ensure_dir(path):
    """
    Ensure directory exists; create recursively if not.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def timestamp():
    """
    Return simple time tag for checkpoint naming/logging.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# ---------------------------------------------------------------
# === Save / Load Functions =====================================
# ---------------------------------------------------------------
def save_checkpoint(df, case, stage, config_name="default_run", base_dir=None, subdir=None):
    """
    Save intermediate or final DataFrame for a given config run.

    Args:
        df : pd.DataFrame
        case : str               -> Case name (e.g., 'Case2')
        stage : str              -> Stage label ('grad1', 'final_rans', etc.)
        config_name : str        -> Identifier for this pipeline config (e.g., 'MLS_u2gdu_v2')
        base_dir : str, optional -> Root output directory
        subdir : str, optional   -> RANS or EXP
    """

    # Default path
    if base_dir is None:
        base_dir = os.path.join("Data", "Shear_mixing", "Processed")

    # Directory structure: Processed/<config_name>/<subdir or case>/
    save_dir = os.path.join(base_dir, config_name)
    if subdir:
        save_dir = os.path.join(save_dir, subdir)
    save_dir = os.path.join(save_dir, case)
    os.makedirs(save_dir, exist_ok=True)

    # Timestamped save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{stage}_{timestamp}.pkl"
    save_path = os.path.join(save_dir, filename)

    df.to_pickle(save_path)
    print(f"[SAVED] {config_name} | {case} | {stage} → {save_path}")
    return save_path


def load_checkpoint(case, stage, base_dir="Processed", suffix="pkl", verbose=True):
    """
    Load previously saved DataFrame checkpoint.
    """
    load_path = Path(base_dir) / case / f"{case}_{stage}.{suffix}"
    if not load_path.exists():
        raise FileNotFoundError(f"[ERROR] Checkpoint not found: {load_path}")

    df = pd.read_pickle(load_path) if suffix == "pkl" else pd.read_csv(load_path)
    if verbose:
        print(f"[LOADED] {case} | {stage} ← {load_path}")
        print(f"         Shape: {df.shape}, Columns: {len(df.columns)}")
    return df


# ---------------------------------------------------------------
# === Utility for Config-driven Paths ===========================
# ---------------------------------------------------------------
def load_case(base_dir, case_name):
    """
    Loads the first .pkl file in `base_dir` that matches the given case name.
    """
    import os
    import pandas as pd

    if not os.path.exists(base_dir):
        print(f"[ERROR] Checkpoint not found: {base_dir}")
        raise FileNotFoundError(f"[ERROR] Base directory not found: {base_dir}")

    candidates = [
        f for f in os.listdir(base_dir)
        if f.endswith(".pkl") and case_name in f
    ]

    if not candidates:
        print(f"[LOADED] {case_name} | 0 candidates")
        raise FileNotFoundError(f"[ERROR] No file found for case {case_name} in {base_dir}")

    file_path = os.path.join(base_dir, candidates[0])
    print(f"[LOADED] {case_name} → {file_path}")
    return pd.read_pickle(file_path)



def list_checkpoints(case, base_dir="Processed"):
    """
    List all checkpoints available for a case.
    """
    case_dir = Path(base_dir) / case
    if not case_dir.exists():
        print(f"[WARN] No directory for {case}")
        return []
    files = sorted(case_dir.glob(f"{case}_*.pkl"))
    for f in files:
        print(f" - {f.name}")
    return files

def detect_output_cols(output_family):
    """
    Return column names for output_family:
    - 'aij'   → anisotropy
    - 'aij_k' → anisotropy + turbulent kinetic energy
    - 'bij'   → normalized Reynolds stress
    - 'rst'   → raw RST
    """
    if output_family == "aij":
        return [f"a_{ij}" for ij in ["xx","xy","xz","yy","yz","zz"]]

    elif output_family == "aij_k":
        # anisotropy + k appended at the end
        return [f"a_{ij}" for ij in ["xx","xy","xz","yy","yz","zz"]] + ["k"]

    elif output_family == "bij":
        return [f"b_{ij}" for ij in ["xx","xy","xz","yy","yz","zz"]]

    elif output_family == "rst":
        # symmetric RST (6 components)
        return ["uu", "uv", "uw", "vv", "vw", "ww"]

    else:
        raise ValueError(f"Unknown output_family '{output_family}'")
