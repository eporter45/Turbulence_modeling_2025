from copy import deepcopy
from Pipelines.Aij.Simulation import main
from Pipelines.Aij.run_sim import cfg
# Define your sweep options
feat_sets =  ['FS1', 'FS2', 'FS3', 'FS4', 'FS8', 'FS7', 'FS6', 'FS5']
norms = ['global', 'column']

# Start sweep loop
for i, feats in enumerate(feat_sets):
    for j, norm in enumerate(norms):
        run_cfg = deepcopy(cfg)   # copy your base cfg each run
        print(f'input: {feats}')
        run_cfg['features']['input'] = feats
        run_cfg['features']['norm'] = norm

        # Give each run a unique name
        run_cfg['paths']['name'] = f"sweep2_{feats}_{norm}"

        print(f"\n[RUNNING] {run_cfg['paths']['name']}")
        main(run_cfg)
