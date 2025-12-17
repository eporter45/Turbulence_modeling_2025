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

# Dynamically resolve project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Trials import TRIALS


def main(config):

    # -------------------------------------------------------------
    #                Base output features (anisotropy)
    # -------------------------------------------------------------
    config['features']['output'] = ['a_xx', 'a_xy', 'a_yy', 'a_xz', 'a_yz', 'a_zz']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    trial_name = config["trial_name"]
    train_cases = TRIALS[trial_name]["train"]
    test_cases = TRIALS[trial_name]["test"]

    print(f"[INFO] Trial: {trial_name}")
    print(f"    Train Cases: {train_cases}")
    print(f"    Test Cases : {test_cases}")

    # -------------------------------------------------------------
    #                  Output Directory Setup
    # -------------------------------------------------------------
    run_name = config["paths"]["name"]
    output_dir = Path(config["paths"]["output_dir"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full config
    import yaml
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    os.environ["KAN_MODEL_DIR"] = str(output_dir / "model")

    # -------------------------------------------------------------
    #                   Load + Normalize Data
    # -------------------------------------------------------------
    data_dir = config["paths"].get("data_dir") or os.path.join(PROJECT_ROOT, 'Data', 'Shear_mixing')

    from PreProcess.load_norm import load_norm
    data_bundle = load_norm(
        config,
        exp_dir=data_dir,
        rans_dir=data_dir,
        save=config['features']['save_kde'],
        save_path=os.path.join(output_dir, "kde")
    )
    print(f'[INFO] Data bundle keys: {data_bundle.keys()}')
    print('[INFO] Finished Load, Norms, and KDE plots')
    print(f'[INFO] Data bundle keys: {list(data_bundle.keys())}')

    # -------------------------------------------------------------
    # Y-norm scaling retrieval
    # -------------------------------------------------------------
    if config['features']['y_norm']:
        yt_max_frob = data_bundle['y_max_frob']
        y_k_max = data_bundle['y_max_k']
        print(f'[INFO] Max y_train Frobenius: {yt_max_frob}')
        print(f'[INFO] Max y_train k: {y_k_max}')
    else:
        yt_max_frob = 1.0
        y_k_max = 1.0

    if not config['features']['denorm_loss']:
        yt_max_frob = 1.0
        y_k_max = 1.0

    # -------------------------------------------------------------
    #                   RUN MODEL (FCN / KAN / TBNN)
    # -------------------------------------------------------------
    model_type = config["model"]["type"].lower()
    print(f'[INFO] Calling Run {model_type} model')

    if model_type == "fcn":
        from Models.FCN import runSimple_model
        fin_preds, fin_model, loss_df, best_model, best_epoch, best_state_dict, best_optimizer = runSimple_model(
            data_bundle, config, output_dir, device,
            y_frob_max=yt_max_frob, y_k_max=y_k_max
        )

    elif model_type == "kan":
        from Models.KAN import runKAN_model
        fin_preds, fin_model, loss_df, best_model, best_epoch, best_state_dict, best_optimizer = runKAN_model(
            data_bundle, config, output_dir, device,
            y_frob_max=yt_max_frob, y_k_max=y_k_max
        )

    elif model_type == "tbnn":
        from Models.TBNN import run_tbnn_model
        fin_preds, fin_model, loss_df, best_model, best_epoch, best_state_dict, best_optimizer = run_tbnn_model(
            config, data_bundle, output_dir, device,
            y_frob_max=yt_max_frob, y_k_max=y_k_max
        )

    else:
        raise NotImplementedError(f"Model type '{model_type}' not supported.")

    # -------------------------------------------------------------
    #                    Save Predictions
    # -------------------------------------------------------------
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    if model_type == 'tbnn':
        fin_pred, fin_coeffs = fin_preds['tensor'], fin_preds['coeffs']
        torch.save(fin_pred, pred_dir / "y_best_pred.pt")
        torch.save(fin_coeffs, pred_dir / "y_best_pred_coeffs.pt")


    else:
        torch.save(fin_preds, pred_dir / "y_best_pred.pt")

    with open(pred_dir / "test_cases.txt", 'w') as f:
        f.writelines([f"{case}\n" for case in test_cases])

    print(f"[INFO] Saved predictions to {pred_dir}")

    # -------------------------------------------------------------
    #                    Truth tensors for metrics
    # -------------------------------------------------------------
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    if model_type == "tbnn":
        # use normalized anisotropy tensor
        y_test = [
            torch.tensor(data_bundle['test'][case]['out_true'] / yt_max_frob, dtype=torch.float32)
            for case in test_cases
        ]
    else:
        # FCN/KAN direct outputs
        y_test = [
            torch.tensor(df.values, dtype=torch.float32)
            for df in data_bundle['y_test'].values()
        ]

    # -------------------------------------------------------------
    #                  Compute & Save Metrics
    # -------------------------------------------------------------
    from Post_Process.make_accuracy_metrics import evaluate_cases

    print("[INFO] Computing test metrics...")
    for tag in ['best', 'fin']:
        evaluate_cases(
            pred_list=fin_preds,
            truth_list=y_test,
            case_names=test_cases,
            config=config,
            output_dir=metrics_dir,
            fin_best=tag
        )

    # -------------------------------------------------------------
    #                  Organize Predictions (Grids)
    # -------------------------------------------------------------
    from Post_Process.organize_results import organize_rst_results
    best_results = organize_rst_results(
        fin_preds, y_test, case_names=test_cases, config=config
    )

    grid_dict = data_bundle['grid_dict']
    lumley_dict = data_bundle['lumley_dict']

    # -------------------------------------------------------------
    #                  Plot Test Predictions
    # -------------------------------------------------------------
    from Plotting.Plot_preds import plot_all_tensor_fields
    plot_all_tensor_fields(
        pred_dict=best_results['pred'],
        truth_dict=best_results['truth'],
        case_names=test_cases,
        grid_dict=grid_dict['test'],
        save_dir=pred_dir,
        config=config,
        best_fin='best'
    )

    # -------------------------------------------------------------
    #                  Plot Lumley (Test)
    # -------------------------------------------------------------
    from Plotting.plot_lumley import plot_lumley_case
    print("[INFO] Plotting Lumley triangles (test cases)...")

    for name in grid_dict['test'].keys():
        plot_lumley_case(
            lumley_dict=lumley_dict,
            bary_preds_by_case=best_results['bary_preds'],
            grid_dict=grid_dict,
            split='test',
            best_fin='Best',
            case=name,
            tol_y=1.5e-3,
            n_pred=20,
            save_dir=pred_dir / 'Lumley_plots'
        )

    # -------------------------------------------------------------
    #                  Truth tensors for TRAIN eval
    # -------------------------------------------------------------
    if model_type == "tbnn":
        y_train = [
            torch.tensor(data_bundle['train'][case]['a_true'] / yt_max_frob, dtype=torch.float32)
            for case in train_cases
        ]
    else:
        y_train = [
            torch.tensor(df.values, dtype=torch.float32)
            for df in data_bundle['y_train'].values()
        ]

    # -------------------------------------------------------------
    #      Optional: Evaluate TRAINING cases (plots + Lumley)
    # -------------------------------------------------------------
    if config.get("eval_training_cases", False):

        print('[INFO] Evaluating Training Cases')

        # Build (inv, tensor_basis) for each case for TBNN
        if model_type == "tbnn":
            x_train_list = []
            for case in train_cases:
                inv = data_bundle['train'][case]['invariants']
                tb  = data_bundle['train'][case]['tensor_basis']

                x_train_list.append((
                    torch.tensor(inv, dtype=torch.float32),
                    torch.tensor(tb, dtype=torch.float32)
                ))
        else:
            x_train_list = [
                torch.tensor(df.values, dtype=torch.float32)
                for df in data_bundle['x_train'].values()
            ]

        from Plotting.plot_train_preds import evaluate_training_cases
        evaluate_training_cases(
            best_model=best_model,
            x_train_list=x_train_list,
            y_train_list=y_train,
            case_names=train_cases,
            grid_dict=grid_dict['train'],
            config=config,
            output_dir=output_dir / "train_predictions",
            model_type=model_type,
            device=device
        )

        print("[INFO] Plotting Lumley triangles (training cases)...")
        for name in grid_dict['train'].keys():
            plot_lumley_case(
                lumley_dict=lumley_dict,
                bary_preds_by_case=best_results['bary_preds'],
                grid_dict=grid_dict,
                split='train',
                best_fin='Best',
                case=name,
                tol_y=1.5e-3,
                n_pred=20,
                save_dir=output_dir / "train_predictions" / "Lumley_plots"
            )

    # -------------------------------------------------------------
    #                        Loss Curves
    # -------------------------------------------------------------
    from Plotting.Plot_loss_histories import plot_all_losses
    print('[INFO] Plotting Loss Histories...')
    plot_all_losses(histories=loss_df, save_dir=output_dir / 'loss_plots')

    # -------------------------------------------------------------
    #      Plot Dynamic Loss Weights (if enabled)
    # -------------------------------------------------------------
    if config['training']['loss'].get('dynamic_weights', False):
        print('[INFO] Plotting Loss Weight History...')
        from train_utils.make_loss_weights import plot_weights_history
        plot_weights_history(
            loss_df,
            save_dir=output_dir / "weights",
            filename="dynamic_component_weights.png"
        )

    print(f"\n✅ [SUCCESS] Finished simulation for {run_name}\n")


# -------------------------------------------------------------
#                   Entry Point
# -------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config)
