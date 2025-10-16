# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 14:43:58 2025

@author: eoporter
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import numpy as np
import pandas as pd
import scipy
import glob
import re

case = 'Case3'
big_data_dir = os.path.join(project_root, 'Data')
save_dir = os.path.join(big_data_dir, 'Shear_mixing','cleaned_exp' )
case_dir = os.path.join(big_data_dir, 'Shear_mixing', case)

def get_nested_dir(case_path, startswith=''):
    for entry in os.listdir(case_path):
        full_path = os.path.join(case_path, entry)
        if (
            os.path.isdir(full_path)
            and entry.startswith(startswith)
            and not entry.startswith("._")
        ):
            return full_path
    raise FileNotFoundError(f"No valid case subdirectory found in {case_path}")

nested_case_dir = get_nested_dir(case_dir, "Case")           # -> Case1_run/
mean_results_dir = get_nested_dir(nested_case_dir, "Mean")   # -> Mean_Results/
data_dir_key = 'Mean' #either Mean or Higher
data_dir = get_nested_dir(mean_results_dir, data_dir_key)          # -> Mean_Side_View_XY/

def load_fovs(data_dir, data_dir_key):
    fov_data = {}
    fov_bounds = {}
    if data_dir_key == 'Higher':
        column_names = [
            "x_mm", "y_mm",
            "uuu", "uvv", "uww", "vuu", "vvv", "vww",
            "uuuu", "vvvv", "wwww"
        ]
    elif data_dir_key == 'Mean':
        column_names = [
            "x_mm", "y_mm", "U", "V", "W", "V_mag",
            "uu", "vv", "ww", "uv"   # updated names
        ]
    else: 
        raise ValueError(f'Invalid data_dir_key used: {data_dir_key}')
    def get_first_data_line_idx(file_path, encoding='cp1252'):
        with open(file_path, 'r', encoding=encoding) as f:
            for i, line in enumerate(f):
                if re.match(r'^\s*[-+]?[0-9]', line):
                    return i

    for filename in os.listdir(data_dir):
        if filename.startswith("._") or not filename.startswith("Side-View"):
            continue

        match = re.search(r"FOV\s+(\d+)\s+\[x\s*=\s*(\d+)-(\d+)\s*mm\]", filename)
        if match:
            fov_num = int(match.group(1))
            x_start = int(match.group(2))
            x_end = int(match.group(3))

            file_path = os.path.join(data_dir, filename)

            # Find the first valid data line
            first_data_row = get_first_data_line_idx(file_path)

            df = pd.read_csv(
                                file_path,
                                sep=r'\s+',
                                skiprows=first_data_row,
                                names=column_names,
                                engine='python',
                                encoding='cp1252'  # TRY THIS
                            )


            fov_data[fov_num] = df
            fov_bounds[fov_num] = (x_start, x_end)

    return fov_data, fov_bounds

print(f'Data dir: {data_dir}')
fov_data, fov_bounds = load_fovs(data_dir, data_dir_key)

for k, v in fov_data.items():
    print(v.columns)



import matplotlib.pyplot as plt

higher_column_names = ["x_mm", "y_mm","uuu", "uvv", "uww", "vuu", "vvv", "vww","uuuu", "vvvv", "wwww"]
mean_column_names = ["x_mm", "y_mm", "U", "V", "W", "V_mag", "uu", "vv", "ww", "uv"]

key = 'V'
#plot_xy_and_U(fov_data[2], key)  # Replace 1 with the FOV number you want to plot


def combine_fovs(fov_data, drop_duplicates=True):
    """
    Combine multiple FOV dataframes into one big dataframe.
    
    Parameters:
    - fov_data: dict of {fov_num: dataframe}
    - drop_duplicates: bool, whether to drop exact duplicate (x_mm, y_mm) points
    
    Returns:
    - combined_df: combined dataframe with all points
    """
    # Concatenate all dataframes
    combined_df = pd.concat(fov_data.values(), ignore_index=True)
    
    if drop_duplicates:
        # Drop exact duplicate spatial points (x_mm, y_mm)
        combined_df = combined_df.drop_duplicates(subset=['x_mm', 'y_mm'])
    
    # Optional: sort by x_mm, then y_mm (just for neatness)
    combined_df = combined_df.sort_values(by=['x_mm', 'y_mm']).reset_index(drop=True)
    
    return combined_df

def save_fovs_as_pickle(fov_data, case_name, save_dir, data_dir_key):
    """
    Save FOV DataFrames as .pkl files using naming format: f{case}_FOV{key}.pkl,
    and save a combined DataFrame as f{case}_combined.pkl

    Args:
        fov_data (dict): Dictionary of FOV DataFrames (key = FOV number)
        case_name (str): The case name string (e.g., "Case1")
        root_dir (str): The root directory path where 'cleaned_exp' will live
    """
    key = ''
    if data_dir_key == 'Mean':
        key = 'mean_flow'
    elif data_dir_key == 'Higher':
        key = 'higher_order'
    else:
        raise ValueError(f'data_dir_key unsupported: {data_dir_key}')
    os.makedirs(save_dir, exist_ok=True)

    combined_df = pd.DataFrame()
    
    for key, df in fov_data.items():
        filename = f"{case_name}_FOV{key}_{data_dir_key}.pkl"
        filepath = os.path.join(save_dir, filename)
        df.to_pickle(filepath)
        print(f"Saved: {filepath}")
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Save combined DataFrame
    combined_filename = f"{case_name}_combined_{data_dir_key}.pkl"
    combined_filepath = os.path.join(save_dir, combined_filename)
    combined_df.to_pickle(combined_filepath)
    print(f"Saved combined DataFrame: {combined_filepath}")
    
#save_fovs_as_pickle(fov_data, case, save_dir, data_dir_key)