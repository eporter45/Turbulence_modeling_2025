# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 11:06:35 2025

@author: eoporter
"""
import os
import sys
import pandas as pd

# Set project root and directory structure
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Define key paths
big_data_dir = os.path.join(project_root, 'Data')
source_dir = os.path.join(big_data_dir, 'Shear_mixing', 'cleaned_exp')
target_dir = os.path.join(big_data_dir, 'Shear_mixing', 'merged_exp')
os.makedirs(target_dir, exist_ok=True)

# Define how many FOVs per case
cases_fov_counts = {
    'Case1': [1, 2, 3, 4],
    'Case2': [1, 2, 3, 4],
    'Case3': [1, 2, 3],
    'Case4': [1, 2, 3],
}

def combine_mean_and_higher(case_name, fov_nums, source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)

    for fov_num in fov_nums:
        mean_file = f"{case_name}_FOV{fov_num}_Mean.pkl"
        higher_file = f"{case_name}_FOV{fov_num}_Higher.pkl"
        
        mean_path = os.path.join(source_dir, mean_file)
        higher_path = os.path.join(source_dir, higher_file)
        
        if not os.path.exists(mean_path) or not os.path.exists(higher_path):
            print(f"[WARNING] Missing files for {case_name} FOV {fov_num}, skipping.")
            continue

        # Load mean and higher order DataFrames
        df_mean = pd.read_pickle(mean_path)
        df_higher = pd.read_pickle(higher_path)

        # Check if (x_mm, y_mm) align exactly
        mean_coords = df_mean[['x_mm', 'y_mm']]
        higher_coords = df_higher[['x_mm', 'y_mm']]

        if not mean_coords.equals(higher_coords):
            print(f"[WARNING] Coordinate mismatch in {case_name} FOV {fov_num}. Doing merge on (x_mm, y_mm).")
            # Merge on coordinates, avoid duplicate coordinate columns from higher order df
            df_merged = pd.merge(df_mean, df_higher.drop(columns=['x_mm', 'y_mm']),
                                 left_on=['x_mm', 'y_mm'], right_index=False, how='inner')
        else:
            # If aligned, just concat columns (excluding duplicate coords)
            df_merged = pd.concat([df_mean, df_higher.drop(columns=['x_mm', 'y_mm'])], axis=1)

        # Save merged dataframe
        merged_filename = f"{case_name}_FOV{fov_num}_data.pkl"
        merged_path = os.path.join(target_dir, merged_filename)
        df_merged.to_pickle(merged_path)
        print(f"Saved merged {case_name} FOV {fov_num} data to: {merged_path}")

# Example usage:
for case, fov_list in cases_fov_counts.items():
    combine_mean_and_higher(case, fov_list, source_dir, target_dir)
