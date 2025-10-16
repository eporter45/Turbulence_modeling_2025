# Turbulence Modeling 2025 Masthead

> Data-driven turbulence closure research unifying high-fidelity flow datasets
> with physics-guided neural networks to improve Reynolds stress transport (RST),
> anisotropy ($a_{ij}$ / $b_{ij}$), and thermodynamic predictions.

The 2025 campaign emphasizes closures that respect turbulence theory while
remaining practical for engineering design loops, blending classical invariants
with modern machine learning pipelines.

Building upon the 2024 pipeline for incompressible turbulence prediction,  
this iteration expands into **compressible flow regimes**, using experimental datasets  
to validate ML-driven models against real physical measurements of shear-layer turbulence.

### 🚀 Project Focus

- **Compressible shear mixing research** focused on modeling and predicting turbulence dynamics in high-speed shear-layer flows.  
- Utilizes the [NASA Langley Compressible Mixing Layer Experimental Dataset](https://turbmodels.larc.nasa.gov/Other_exp_Data/compressible_mixinglayers_exp.html) as the primary benchmark for validation and feature development.  
- Investigates **anisotropy, turbulent kinetic energy transport, and compressibility effects** within shear-driven turbulence.  
- Develops and evaluates **physics-guided neural networks** and hybrid ML–CFD approaches for reconstructing flow-field quantities and closure terms.  
- Emphasizes **data-driven generalization across Mach regimes**, enabling models to learn physical correlations between density gradients, Reynolds stresses, and energy dissipation in compressible turbulence.


### Primary Maintainers
- **Elliot Porter** – original author of the simulation pipelines and trial
  definitions (`Pipelines/RST/run_sim.py`).
- **Research Applications Team** – curates datasets under `Data/` and develops
  preprocessing utilities in `PreProcess/` and `Features/`.

### Technical Pillars
| Pillar | Description |
| --- | --- |
| Experiment orchestration | `Pipelines/` contains runnable scripts for FCN and KAN model families with configurable schedulers and loss weights. |
| Physics-informed training | `Supported_loss_types` defines allowable data, physics, and constraint loss aliases to enforce turbulence theory. |
| Reusable utilities | `train_utils/` exposes helper functions for feature indexing, tensor reshaping, and loss scaling across experiments. |

### Outputs and Documentation
- Trained checkpoints, logs, and diagnostic plots are collected under
  `outputs/` and the per-pipeline `Outputs/` folders for downstream analysis and
  reporting.
- Reference `Code_documentation.docx` for extended background notes and
  rationale behind architectural decisions.

---

## Repository Layout

The project is organized around a few key themes:

- **`Data/`** – Raw and intermediate data products, including historic exports
  under `Old_data/` and the current shear-mixing campaign under
  `Shear_mixing/`. 【F:Data/Shear_mixing†L1-L1】
- **`Features/`** – Feature engineering helpers and scripts for assembling
  machine-learning-ready inputs. 【F:Features†L1-L1】
- **`PreProcess/`** – Modules that resolve paths, normalize signals, and compute
  classical turbulence invariants prior to training. 【F:PreProcess/Load_and_norm.py†L1-L160】
- **`Pipelines/`** – End-to-end experiment entry points. Separate sub-folders
  exist for RST, anisotropy, thermodynamic, and all-target sweeps. The RST
  pipeline (`Pipelines/RST/run_sim.py`) demonstrates how trials, model choices,
  schedulers, and custom losses are composed. 【F:Pipelines/RST/run_sim.py†L1-L120】
- **`train_utils/`** – Shared training building blocks such as tensor reshaping
  helpers, physical loss calculators, weighting utilities, and schedulers.
  Example: `helpers.py` reconstructs Reynolds stress tensors from predicted
  components. 【F:train_utils/helpers.py†L1-L120】
- **`Supported_loss_types`** – Registry of recognized data, physics, and
  constraint loss aliases used when configuring experiments.
  【F:Supported_loss_types†L1-L32】
- **`Post_Process/` and `Plotting/`** – Analysis notebooks and plotting scripts
  for evaluating trained models.
- **`Models/`** – Archived model weights and experiment artifacts.

## Getting Started

1. **Create an environment.** The code targets Python 3.10+ with `torch` for
   deep learning. Create and activate a virtual environment, then install
   dependencies, e.g.

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   *(If a `requirements.txt` file is not yet available, consult the pipeline
   scripts to gather the necessary packages manually.)*

2. **Configure data paths.** Update the `paths` section in your experiment
   configuration (see `Pipelines/RST/run_sim.py`) so that `data_dir` points to
   the location of your processed feature tensors or raw flow exports.

3. **Select a trial definition.** Training and evaluation case splits are
   defined in `Trials.py`. Choose an entry (e.g. `single_inter_c2_f2`) and
   reference it via the `trial_name` field in your pipeline configuration.
   【F:Trials.py†L1-L47】

4. **Run a simulation.** Each pipeline provides a launch script. For example,
   execute an RST experiment with:

   ```bash
   python Pipelines/RST/run_sim.py
   ```

   The script resolves project paths, seeds the random number generators, builds
   a model configuration (FCN or KAN), and starts training with the specified
   loss schedule.

5. **Monitor outputs.** Results are written to the directory specified by
   `paths.output_dir` (defaults to `Pipelines/RST/outputs/`). Inspect the saved
   checkpoints, logs, and plots to assess model performance.

## Customizing Experiments

- **Loss design.** Toggle physics and constraint losses via the `training.loss`
  configuration block. Valid aliases are documented in
  `Supported_loss_types`. 【F:Supported_loss_types†L1-L32】
- **Feature families.** The helper utilities can automatically detect Reynolds
  stress, anisotropy, and turbulence kinetic energy outputs, enabling metrics
  such as Reynolds tensor reconstruction. 【F:train_utils/helpers.py†L1-L97】
- **Normalization.** The preprocessing modules (`PreProcess/normalize_data.py`,
  etc.) allow dimensional or non-dimensional scaling; ensure your selections
  match the `features` block of the pipeline configuration.

## Documentation

Additional background material is captured in `Code_documentation.docx` and the
notes embedded throughout the pipeline scripts. Use these alongside the README
when onboarding new contributors or preparing new studies.

## Contributing

1. Fork the repository and create a feature branch.
2. Make your changes along with tests or demonstration notebooks.
3. Run the relevant pipeline scripts to confirm regressions are avoided.
4. Submit a pull request summarizing your contribution and results.

---

For questions about experimental design or dataset provenance, start by reading
through the trial definitions in `Trials.py` and the inline comments in the
pipeline configuration files. These outline the intended flow cases, feature
sets, and training assumptions for the 2025 campaign.
