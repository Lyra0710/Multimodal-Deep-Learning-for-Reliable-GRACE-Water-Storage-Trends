# Multimodal-Deep-Learning-for-Reliable-GRACE-Water-Storage-Trends
# DATA 255 — Group 5 Project

**Project:** Multimodal Deep Learning for Reliable GRACE Water Storage Trends

## Project Overview

This project investigates whether multimodal deep-learning models can reliably predict changes in terrestrial water storage (TWS) using NASA GRACE/GRACE-FO observations, precipitation, and temperature data.

We aim to compare different forecasting approaches, analyze water-storage trends, and evaluate model reliability across different environmental conditions.

### Current Scope: Intermediate Project

For the intermediate submission, we will focus on:

- Data collection and preprocessing
- Exploratory Data Analysis (EDA)
- Statistical trend analysis
- Baseline model implementation and evaluation
- Literature review

Advanced deep-learning models (LSTM, TCN, Transformer), multimodal fusion experiments, uncertainty estimation, and explainability will be addressed in the final project.

---

## GRACE-FO Download, Preprocessing, and Persistence Baseline

Create a local `.env` file and add your NASA Earthdata credentials. Do not commit `.env`.

```bash
cp .env.example .env
```

Run the processing script from the repository root to download (or safely reuse) the original NASA source in `data/raw/` and write separate, validated GRACE-FO outputs to `data/processed/`:

```bash
python scripts/grace_fo_processing.py \
  --download-dir data/raw \
  --output data/processed/grace_fo_processed.nc \
  --land-view-output data/processed/grace_fo_land_modeling_view.nc \
  --env-file .env
```

The script preserves the raw NASA NetCDF unchanged. A `SKIPPED` download message means the existing source file was reused successfully. The expected validation output covers 84 monthly observations from January 2019 through December 2025 on a 360 × 720 grid. The full `grace_fo_processed.nc` retains all NASA variables; `grace_fo_land_modeling_view.nc` is a separate optional land-only view.

Run the reproducible one-month-ahead persistence baseline with the shared chronological splits:

```bash
python scripts/grace_fo_persistence.py \
  --input data/processed/grace_fo_processed.nc \
  --config configs/grace_fo_config.yaml \
  --metrics-dir results/metrics \
  --figures-dir results/figures \
  --split-output-dir data/processed \
  --model-output models/grace_fo_one_month_persistence_baseline.json
```

It saves `grace_fo_persistence_metrics.json`, `grace_fo_chronological_splits.csv`, `grace_fo_one_month_persistence_baseline.json`, and one GRACE-FO actual-versus-predicted figure per split. The split manifest is deterministic and uses no random seed. The saved model specification records the persistence equation and confirms that it has no learned parameters. See [grace_fo_eda.md](grace_fo_eda.md) for the existing notebook's EDA findings, preprocessing decisions, and limitations.

### Current GRACE-FO artifacts and Member 4 handoff

Member 4 should directly load these ready-to-use chronological GRACE-FO target-month datasets:

- `data/processed/grace_fo_train.nc`: January 2019–December 2022 target months.
- `data/processed/grace_fo_validation.nc`: January–December 2023 target months.
- `data/processed/grace_fo_test.nc`: January 2024–December 2025 target months.

Each split retains the native 0.5-degree grid and all needed NASA variables: `lwe_thickness`, `uncertainty`, `land_mask`, `mascon_ID`, `scale_factor`, `GAD`, coordinate bounds, and `original_time`. The split files are deterministic chronological subsets—no random seed or random sampling was used.

Supporting GRACE-FO artifacts are:

- `configs/grace_fo_config.yaml`: fixed chronological target-month boundaries—train (2019–2022), validation (2023), and test (2024–2025).
- `results/metrics/grace_fo_chronological_splits.csv`: direct month-by-month split membership, prior persistence input month, and persistence eligibility. It is deterministic; no random seed was used.
- `results/metrics/grace_fo_persistence_metrics.json`: one-month persistence baseline metrics.
- `models/grace_fo_one_month_persistence_baseline.json`: the saved deterministic model specification.
- `results/figures/grace_fo_persistence_*_actual_vs_predicted.png`: baseline actual-versus-predicted figures.

`data/processed/grace_fo_processed.nc` remains the full processed master for reproducibility and pipeline reruns. Member 4 does not need the raw NASA NetCDF or the optional `grace_fo_land_modeling_view.nc`. For final uncertainty and explainability work, Member 4 will also need the eventual model predictions and integrated environmental inputs from the relevant team members.

---

## Folder Guidelines

### `data/`

Contains datasets used throughout the project.

- `raw/`: Original downloaded NASA datasets (GRACE/GRACE-FO, GPM IMERG, MERRA-2).
- `processed/`: Cleaned, standardized, and merged datasets ready for analysis and modeling.

**Important:** Do not commit raw NASA downloads. The designated GRACE-FO handoff artifacts are `data/processed/grace_fo_train.nc`, `data/processed/grace_fo_validation.nc`, and `data/processed/grace_fo_test.nc`; keep other large duplicate outputs in shared storage and document their sources and download instructions.

### `scripts/`

Contains reusable Python scripts for data preprocessing, analysis, modeling, and evaluation.

Each member should place their implementation in the relevant `.py` files.

Keep scripts organized and avoid duplicating functionality already implemented by another member.

### `notebooks/`

Contains Jupyter notebooks for exploratory analysis, experiments, and visualizations.

Each member should primarily work in their assigned notebook.

Use notebooks to demonstrate the analysis and results. Move reusable functions into `scripts/` whenever practical.

### `models/`

Contains trained model files, such as `.pkl` or `.joblib`.

Large model files should not be committed to GitHub.

### `results/`

Contains outputs generated from the experiments.

- `figures/`: EDA plots, trend maps, actual-vs-predicted plots, and other visualizations.
- `metrics/`: Model evaluation results, including MAE, RMSE, correlation, and comparison tables.

Use descriptive filenames, such as `grace_missing_months.png` or `persistence_metrics.csv`.

### `reports/`

Contains written documentation and report materials.

Each member is responsible for individually reviewing at least 13 relevant research papers for the intermediate submission. Use APA citation format and write the survey in your own words.

---

## Revanth: combined-data analysis and Ridge baseline

See [the run guide and current status](reports/revanth_combined_ridge.md) and
[the companion notebook](notebooks/Revanth_Combined_Analysis_Ridge.ipynb).
The implementation expects Rachan's `aligned_0.5deg.nc` handoff and uses the shared
chronological split configuration. Real-data results are pending that input;
synthetic tests are software checks only.
