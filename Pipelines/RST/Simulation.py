# -*- coding: utf-8 -*-
"""
RST simulation driver script
Updated to match modern pipeline (2025).
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

    # Output fields = Reynolds stresses (upper triangular vector6)
    config['features']['output'] = ['uu', 'uv', 'vv', 'uw', 'vw', 'ww']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    trial_name = config["trial_name"]
    assert trial_name in TRIALS, f"Trial '{trial_name}' not found in TRIALS"

    train_cases = TRIALS[trial_name]["train"]
    test_cases  = TRIALS[trial_name]["test"]

    print(f"[INFO] Trial: {trial_name}")
    print(f"      Train Cases: {train_cases}")
    print(f"      Test Cases : {test_cases}")

    # ---------------------------------------------------------------
    # Setup output directory
    # ---------------------------------------------------------------
    run_name = config["paths"]["name"]
    output_dir = Path(config["paths"]["output_dir"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    import yaml
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    os.environ["KAN_MODEL_DIR"] = str(output_dir / "model")

    # ---------------------------------------------------------------
    # Load + normalize
    # ---------------------------------------------------------------
    data_dir = config["paths"].get("data_dir") or os.path.join(PROJECT_ROOT, "Data", "Shear_mixing")

    exp_dir  = os.path.join(data_dir, "EXP", "train_exp")
    rans_dir = os.path.join(data_dir, "RANS", "training")

    from PreProcess.load_norm import load_norm
    data_bundle = load_norm(
        config,
        exp_dir=exp_dir,
        rans_dir=rans_dir,
        save=config['features']['save_kde'],
        save_path=output_dir / "KDE"
    )

    print("[INFO] Finished Load + Norm + KDE")

    # Extract structured items
    x_train_dict = data_bundle["x_train"]
    x_test_dict  = data_bundle["x_test"]

    x_train_norm = data_bundle["x_train_normed"]
    x_test_norm  = data_bundle["x_test_normed"]

    y_train_dict = data_bundle["y_train_normed"]
    y_test_dict  = data_bundle["y_test_normed"]

    grid_dict   = data_bundle["grid_dict"]
    lumley_dict = data_bundle["lumley_dict"]

    # ---------------------------------------------------------------
    # Y normalization scaling
    # ---------------------------------------------------------------
    if config['features']['y_norm']:
        yt_max_frob = data_bundle['y_max_frob']
        y_k_max     = data_bundle['y_max_k']
        print(f"[INFO] Max y Frobenius = {yt_max_frob}")
        print(f"[INFO] Max y k         = {y_k_max}")
    else:
        yt_max_frob = 1.0
        y_k_max     = 1.0

    if not config['features']['denorm_loss']:
        yt_max_frob = 1.0
        y_k_max     = 1.0

    # ---------------------------------------------------------------
    # Train FCN or KAN model
    # ---------------------------------------------------------------
    model_type = config["model"]["type"].lower()
    print(f"[INFO] Running model type: {model_type}")

    # Build ordered lists (FCN/KAN expect list of tensors)
    x_train_list = [
        torch.tensor(x_train_norm[c].values, dtype=torch.float32)
        for c in train_cases
    ]
    y_train_list = [
        torch.tensor(y_train_dict[c].values, dtype=torch.float32)
        for c in train_cases
    ]
    x_test_list = [
        torch.tensor(x_test_norm[c].values, dtype=torch.float32)
        for c in test_cases
    ]
    y_test_list = [
        torch.tensor(y_test_dict[c].values, dtype=torch.float32)
        for c in test_cases
    ]

    if model_type == "fcn":
        from Models.FCN import runSimple_model
        fin_pred, y_pred_best, loss_df, best_model, best_epoch, best_state_dict, best_optimizer = runSimple_model(
            x_train_list,
            y_train_list,
            x_test_list,
            config,
            output_dir,
            device,
            y_frob_max=yt_max_frob,
            y_k_max=y_k_max
        )

    elif model_type == "kan":
        from Models.KAN import runKAN_model
        fin_pred, y_pred_best, loss_df, best_model, best_epoch, best_state_dict, best_optimizer = runKAN_model(
            x_train_list,
            y_train_list,
            x_test_list,
            config,
            output_dir,
            device,
            y_frob_max=yt_max_frob,
            y_k_max=y_k_max
        )

    else:
        raise NotImplementedError(f"Model type '{model_type}' not supported for RST simulation.")

    # ---------------------------------------------------------------
    # Save predictions
    # ---------------------------------------------------------------
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    torch.save(y_pred_best, pred_dir / "y_best_pred.pt")
    torch.save(fin_pred,  pred_dir / "y_fin_pred.pt")
    torch.save(y_test_list, pred_dir / "y_test.pt")

    with open(pred_dir / "test_cases.txt", 'w') as f:
        f.writelines([c + "\n" for c in test_cases])

    print(f"[INFO] Saved predictions → {pred_dir}")

    # ---------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    from Post_Process.make_accuracy_metrics import evaluate_cases

    evaluate_cases(
        pred_list=y_pred_best,
        truth_list=y_test_list,
        case_names=test_cases,
        config=config,
        output_dir=metrics_dir,
        fin_best="best"
    )

    evaluate_cases(
        pred_list=fin_pred,
        truth_list=y_test_list,
        case_names=test_cases,
        config=config,
        output_dir=metrics_dir,
        fin_best="fin"
    )

    # ---------------------------------------------------------------
    # Organize results (grid-shaped tensors)
    # ---------------------------------------------------------------
    from Post_Process.organize_results import organize_rst_results
    best_results = organize_rst_results(y_pred_best, y_test_list, case_names=test_cases, config=config)
    fin_results  = organize_rst_results(fin_pred,  y_test_list, case_names=test_cases, config=config)

    # ---------------------------------------------------------------
    # Prediction plots
    # ---------------------------------------------------------------
    from Plotting.Plot_preds import plot_all_tensor_fields

    plot_all_tensor_fields(
        best_results['pred'],
        best_results['truth'],
        case_names=test_cases,
        grid_dict=grid_dict['test'],
        save_dir=pred_dir,
        config=config,
        best_fin="best"
    )

    plot_all_tensor_fields(
        fin_results['pred'],
        fin_results['truth'],
        case_names=test_cases,
        grid_dict=grid_dict['test'],
        save_dir=pred_dir,
        config=config,
        best_fin="fin"
    )

    # ---------------------------------------------------------------
    # Lumley plots
    # ---------------------------------------------------------------
    from Plotting.plot_lumley import plot_lumley_case

    for name in grid_dict['test']:
        plot_lumley_case(
            lumley_dict=lumley_dict,
            bary_preds_by_case=best_results['bary_preds'],
            grid_dict=grid_dict,
            split='test',
            best_fin='Best',
            case=name,
            tol_y=1.5e-3,
            n_pred=20,
            save_dir=pred_dir / "Lumley_plots"
        )
        plot_lumley_case(
            lumley_dict=lumley_dict,
            bary_preds_by_case=fin_results['bary_preds'],
            grid_dict=grid_dict,
            split='test',
            best_fin='Final',
            case=name,
            tol_y=1.5e-3,
            n_pred=20,
            save_dir=pred_dir / "Lumley_plots"
        )

    # ---------------------------------------------------------------
    # Optional: Training-set prediction plots
    # ---------------------------------------------------------------
    if config.get("eval_training_cases", False):

        from Plotting.plot_train_preds import evaluate_training_cases

        # FCN/KAN input format
        x_train_eval = [
            torch.tensor(df.values, dtype=torch.float32)
            for df in x_train_norm.values()
        ]
        y_train_eval = [
            torch.tensor(df.values, dtype=torch.float32)
            for df in y_train_dict.values()
        ]

        evaluate_training_cases(
            best_model=best_model,
            x_train_list=x_train_eval,
            y_train_list=y_train_eval,
            case_names=train_cases,
            grid_dict=grid_dict['train'],
            config=config,
            output_dir=output_dir / "train_predictions",
            model_type=model_type,
            device=device
        )

    # ---------------------------------------------------------------
    # Loss history plots
    # ---------------------------------------------------------------
    from Plotting.Plot_loss_histories import plot_all_losses
    print("[INFO] Plotting Loss Histories...")
    plot_all_losses(loss_df, save_dir=output_dir / "loss_plots")

    # ---------------------------------------------------------------
    # Plot dynamic weights (if enabled)
    # ---------------------------------------------------------------
    if config['training']['loss'].get('dynamic_weights', False):
        from train_utils.make_loss_weights import plot_weights_history
        plot_weights_history(
            loss_df,
            save_dir=output_dir / "weights",
            filename="dynamic_component_weights.png"
        )

    print(f"\n✅  [SUCCESS] Finished RST simulation for {run_name}\n")


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------
if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)
