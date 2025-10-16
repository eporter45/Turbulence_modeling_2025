# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 23:06:40 2025

@author: eoporter
"""
import os
def resolve_config_paths(config, project_root):
    """Resolve important relative paths in the config using project_root."""
    
    # Only update if path is relative or empty
    def resolve(path, *subdirs):
        if not path or not os.path.isabs(path):
            return os.path.join(project_root, *subdirs)
        return path

    # Resolve main paths
    config["paths"]["data_dir"] = resolve(config["paths"].get("data_dir", ""), "Data", "Shear_mixing")
    config["paths"]["output_dir"] = resolve(config["paths"].get("output_dir", ""), "outputs")

    return config