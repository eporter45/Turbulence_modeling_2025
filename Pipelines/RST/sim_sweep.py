import os
import random
import numpy as np
import torch
from PreProcess.resolve_root import resolve_config_paths
from Pipelines.RST.run_sim import cfg



def get_base_config(config, PROJECT_ROOT):
    

    config = resolve_config_paths(config, PROJECT_ROOT)

    # Set random seeds
    seed = cfg['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    return cfg


# Resolve PROJECT_ROOT
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Pipelines.RST.Simulation import main
from Pipelines.RST.configs.base_config import get_base_config


def run_sweep():
    # sweep params
    

        # Set a unique trial name for saving results
        cfg["trial_name"] = overrides.get("trial_name", f"sweep_case_{i}")
        cfg["paths"]["name"] = cfg["trial_name"]

        main(cfg)


if __name__ == "__main__":
    run_sweep()