# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 09:41:05 2025

@author: eoporter
"""

import argparse
from pathlib import Path
import torch
import os
import sys
#Dynamically resolve project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add to system path if not already
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from Trials import TRIALS

'''
aij --> t_ij 
aij -- > b_ij
both by using tke. 

'''


def main(config):
    config['features']['output'] = ['a_xx', 'a_xy', 'a_yy', 'a_xz', 'a_yz', 'a_zz', 'tke']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    trial_name = config["trial_name"]
    #assert trial_name in TRIALS, f"Trial '{trial_name}' not found in TRIALS"
    train_cases = TRIALS[trial_name]["train"]
    test_cases = TRIALS[trial_name]["test"]

    print(f"[INFO] Trial: {trial_name}")
    print(f"    Train Cases: {train_cases}")
    print(f"    Test Cases : {test_cases}")

    # Set up output dir
    run_name = config["paths"]["name"]
    output_dir = Path(config["paths"]["output_dir"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    import yaml
   # Save config.yaml inside the output_dir
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    os.environ["KAN_MODEL_DIR"] = str(output_dir / "model")

    data_dir = config["paths"].get("data_dir")
    if not data_dir:
        data_dir = os.path.join(PROJECT_ROOT, 'Data', 'Shear_mixing')
    exp_dir = os.path.join(data_dir,'EXP', 'train_exp')
    rans_dir = os.path.join(data_dir, 'RANS', 'training')
    
    from PreProcess.Load_and_norm import load_norm_kde
    data_bundle = load_norm_kde(config, exp_dir=exp_dir, rans_dir=rans_dir, 
                                save=config['features']['save_kde'],
                                save_path= output_dir/'KDE')
    print('[INFO] Finished Load, Norms and KDE plots')
    x_train = data_bundle["x_train"]
    y_train = data_bundle["y_train_normed"] #still same if norm turned off
    x_test = data_bundle["x_test"]
    y_test = data_bundle["y_test_normed"]
    x_train_normed = data_bundle["x_train_normed"]
    x_test_normed = data_bundle["x_test_normed"]
    grid_dict = data_bundle["grid_dict"]
    lumley_dict = data_bundle['lumley_dict']
    if config['features']['y_norm']:
        yt_max_frob = data_bundle['y_max_frob']
        y_k_max = data_bundle['y_max_k']
        print(f'[INFO] Max y_train frobenius: {yt_max_frob}')
        print(f'[INFO] Max y_train k: {y_k_max}')
    if not config['features']['denorm_loss']:
        yt_max_frob = 1.0
        y_k_max = 1.0
    # === Model Training ===
    model_type = config["model"]["type"].lower()
    y_pred = None
    print(f'[INFO] Calling Run {model_type} model')
    if model_type == "fcn":
        from Models.FCN import runSimple_model
        #    return predictions, loss_df, best_model, best_epoch, best_state_dict, optimizer
        fin_pred, y_pred, loss_df, best_model, best_epoch, best_state_dict, best_optimizer = runSimple_model(
            x_train_normed, y_train, x_test_normed, config, output_dir, device, y_frob_max=yt_max_frob, y_k_max=y_k_max
        )
    elif model_type == "kan":
        from Models.KAN import runKAN_model
        fin_pred, y_pred, loss_df, best_model, best_epoch, best_state_dict, best_optimizer = runKAN_model(
            x_train_normed, y_train, x_test_normed, config, output_dir, device, y_frob_max = yt_max_frob, y_k_max=y_k_max
        )
    else:
        raise NotImplementedError(f"Model type '{model_type}' not supported.")

    # === Inverse Transform if needed ===

    # === Save Predictions ===
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    torch.save(y_pred, pred_dir / "y_best_pred.pt")
    torch.save(fin_pred, pred_dir/'y_fin_pred.pt')
    torch.save(y_test, pred_dir / "y_test.pt")

    with open(pred_dir / "test_cases.txt", 'w') as f:
        f.writelines([f"{case}\n" for case in test_cases])

    print(f"[INFO] Saved predictions to {pred_dir}")

    # === Compute Test Losses ===
    # Call test loss computation
    metrics_dir = output_dir / "metrics"

    # === Metrics ===
    from Post_Process.make_accuracy_metrics import evaluate_cases
     # Run metrics evaluation across all test cases
    evaluate_cases(
        pred_list=y_pred,              # list of [N, 6] tensors
        truth_list=y_test,             # list of [N, 6] tensors
        case_names=test_cases,         # list of strings
        config=config,                 
        output_dir=metrics_dir,
        fin_best='best')
    evaluate_cases(
        pred_list=fin_pred,              # list of [N, 6] tensors
        truth_list=y_test,             # list of [N, 6] tensors
        case_names=test_cases,         # list of strings
        config=config,                 
        output_dir=metrics_dir,
        fin_best = 'fin')
    # aggregated metrics (already saved per-case inside evaluate_cases)
    test_names = list(grid_dict['test'].keys())
    from Post_Process.organize_results import organize_rst_results
    best_results = organize_rst_results(y_pred, y_test,case_names=test_names,config= config)
    fin_results = organize_rst_results(fin_pred, y_test,case_names=test_names, config= config)
    for key in best_results['pred']:
        print(f"{key}: {[p.shape for p in best_results['pred'][key]]}")
     
        
    # === Plot Predictions ===
    from Plotting.Plot_preds import plot_all_tensor_fields
    plot_all_tensor_fields(best_results['pred'],
                            best_results['truth'],
                            case_names=test_cases,
                            save_dir=output_dir/'predictions',
                            grid_dict=grid_dict['test'],
                            config=config,
                            best_fin='best'
                        )
    plot_all_tensor_fields(fin_results['pred'],
                            fin_results['truth'],
                            case_names=test_cases,
                            save_dir=output_dir/'predictions',
                            grid_dict=grid_dict['test'],
                            config=config,
                            best_fin='fin'
                        )
    from Plotting.plot_lumley import plot_lumley_case
    for name in test_names:
        plot_lumley_case(
            lumley_dict = lumley_dict,
            bary_preds_by_case = best_results['bary_preds'],  # <-- now keyed by case
            grid_dict   = grid_dict,
            split       = 'test',
            best_fin = 'Best',
            case        = name,
            tol_y       = 1.5e-3,
            n_pred      = 20,
            save_dir    = output_dir/'predictions'/'Lumley_plots')  # wherever you want files
        plot_lumley_case(
            lumley_dict = lumley_dict,
            bary_preds_by_case = fin_results['bary_preds'],  # <-- now keyed by case
            grid_dict   = grid_dict,
            split       = 'test',
            best_fin = 'Final',
            case        = name,
            tol_y       = 1.5e-3,
            n_pred      = 20,
            save_dir    = output_dir/'predictions'/'Lumley_plots')  # wherever you want files
    
    
    #===== If enabeled plot predictions on training set========
    if config.get("eval_training_cases", False):
        from Plotting.plot_train_preds import evaluate_training_cases
        print('[INFO] Evaluating Training Cases')
        evaluate_training_cases(
            best_model=best_model,
            x_train_list=x_train,
            y_train_list=y_train,
            case_names=train_cases,
            grid_dict=grid_dict['train'],
            config=config,
            output_dir= output_dir/"train predictions" # same as where test plots/metrics go
        )
        train_names = list(grid_dict['train'].keys())
        for name in train_names:
            plot_lumley_case(
                lumley_dict = lumley_dict,
                bary_preds_by_case = best_results['bary_preds'],  # <-- now keyed by case
                grid_dict   = grid_dict,
                split       = 'train',
                best_fin = 'Best',
                case        = name,
                tol_y       = 1.5e-3,
                n_pred      = 20,
                save_dir    = output_dir/'train_predictions'/'Lumley_plots')  # wherever you want files
            plot_lumley_case(
                lumley_dict = lumley_dict,
                bary_preds_by_case = fin_results['bary_preds'],  # <-- now keyed by case
                grid_dict   = grid_dict,
                split       = 'train',
                best_fin = 'Final',
                case        = name,
                tol_y       = 1.5e-3,
                n_pred      = 20,
                save_dir    = output_dir/'train_predictions'/'Lumley_plots')  # wherever you want files
    #=======Plot loss histories =======
    from Plotting.Plot_loss_histories import plot_all_losses
    print('[INFO] Plotting Loss Histories')

    plot_all_losses(
        histories=loss_df,
        save_dir= output_dir / 'loss_plots'
            )
    
    #=======If Dynamic weights are enebeled Plott them =======
    if config['training']['loss'].get('dynamic_weights', False):
        print('[INFO] Plotting Loss Weight History')
        from train_utils.make_loss_weights import plot_weights_history
        plot_weights_history(loss_df,  # or some stored variable
                            save_dir=output_dir / "weights",
                            filename="dynamic_component_weights.png")

    

    print(f"[SUCCESS] Finished simulation for {run_name}")


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config)