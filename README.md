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

Run the following steps from the repository root, in order. They reproduce the GRACE-FO persistence artifacts used by this project. The notebook `notebooks/GRACE_GRACE_FO_Data_Analysis.ipynb` follows the same order when run top to bottom.

1. Install the GRACE workflow dependencies once:

```bash
python -m pip install numpy pandas xarray matplotlib PyYAML netCDF4 python-dotenv podaac-data-subscriber
```

2. Download (or safely reuse) the original NASA source in `data/raw/`, then write the validated processed master and optional land view:

```bash
python scripts/grace_fo_processing.py \
  --download-dir data/raw \
  --output data/processed/grace_fo_processed.nc \
  --land-view-output data/processed/grace_fo_land_modeling_view.nc \
  --env-file .env
```

The script preserves the raw NASA NetCDF unchanged. A `SKIPPED` download message means the existing source file was reused successfully. The expected validation output covers 84 monthly observations from January 2019 through December 2025 on a 360 × 720 grid. The full `grace_fo_processed.nc` retains all NASA variables; `grace_fo_land_modeling_view.nc` is a separate optional land-only view.

3. Run the reproducible one-month-ahead persistence baseline with the shared chronological splits:

```bash
python scripts/grace_fo_persistence.py \
  --input data/processed/grace_fo_processed.nc \
  --config configs/grace_fo_config.yaml \
  --metrics-dir results/metrics \
  --figures-dir results/figures \
  --split-output-dir data/processed \
  --model-output models/grace_fo_one_month_persistence_baseline.json
```

It saves `grace_fo_persistence_metrics.json`, `grace_fo_chronological_splits.csv`, `grace_fo_one_month_persistence_baseline.json`, and one GRACE-FO actual-versus-predicted figure per split. The split manifest is deterministic and uses no random seed. The saved model specification records the persistence equation and confirms that it has no learned parameters.

Persistence evaluation uses only `land_mask == 1` cells from 60°S through 60°N. The primary `rmse_cm` is the cosine-latitude area-weighted average of per-cell temporal RMSE values; `pooled_rmse_cm` is retained as the legacy pooled cell-month measure. The fixed target-month splits are train (2019-01-01–2022-12-31), validation (2023-01-01–2023-12-31), and test (2024-01-01–2025-12-31). The test output contains 24 target-month coordinates from 2024-01-01 through 2025-12-01.

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

## Precipitation & Temperature Drivers of TWS and Climate-Driven Baselines

This stage puts GRACE/GRACE-FO TWS (2002-04 onward), GPM IMERG precipitation, and MERRA-2 2 m air temperature on one 0.5° grid and calendar. It then analyzes P/T trends and their relationship with TWS, and evaluates the baselines the deep-learning models must beat. Train/validation/test dates are read from `configs/grace_fo_config.yaml`, so these baselines use exactly the same months as the persistence baseline. All other settings live in `configs/p_t_tws_config.yaml`.

The GRACE CRI NetCDF is reused from `data/raw/` (or downloaded with `grace_fo_processing.py`'s downloader). IMERG and MERRA-2 are streamed with `earthaccess` using the same `.env` credentials. For these GES DISC products, also authorize the *NASA GESDISC DATA ARCHIVE* application in your Earthdata profile. Reduced driver cubes are cached in `data/processed/p_t_tws_cache/`; the first run takes about an hour per product.

```bash
python -m pip install earthaccess pymannkendall h5netcdf scikit-learn joblib

python scripts/p_t_tws_data.py \
  --config configs/p_t_tws_config.yaml \
  --output data/processed/p_t_tws_aligned_0.5deg.nc \
  --env-file .env

python scripts/p_t_tws_analysis.py \
  --input data/processed/p_t_tws_aligned_0.5deg.nc \
  --config configs/p_t_tws_config.yaml \
  --metrics-dir results/metrics \
  --figures-dir results/figures \
  --maps-dir data/processed

python scripts/p_t_tws_baselines.py \
  --input data/processed/p_t_tws_aligned_0.5deg.nc \
  --config configs/p_t_tws_config.yaml \
  --metrics-dir results/metrics \
  --figures-dir results/figures \
  --predictions-output data/processed/p_t_tws_baseline_predictions.nc \
  --model-output models/p_t_tws_baselines.json
```

[notebooks/P_T_TWS_Drivers_and_Baselines.ipynb](notebooks/P_T_TWS_Drivers_and_Baselines.ipynb) runs the same steps interactively, with the methodology explained per section.

- Anomalies use a monthly climatology fitted on the training split only. Pixels are land (`land_mask`), |lat| ≤ 60°, finite P/T, and ≥ 70% of GRACE months observed.
- Trends: Mann–Kendall with Hamed–Rao correction, Theil–Sen slopes, and Benjamini–Hochberg FDR. Correlations use an autocorrelation-adjusted effective sample size.
- Temporal baselines: climatology, persistence, seasonal persistence, per-pixel ridge (seasonal + trend, P/T drivers, ARX), and pooled gradient boosting on seasonal-trend residuals. Hyperparameters are chosen on validation, models are refit on train + validation, and all are scored once on identical test cells.
- Geographic baselines: pooled ridge and gradient boosting under 10° × 10° spatial block cross-validation.

Outputs:

- `data/processed/p_t_tws_aligned_0.5deg.nc`: aligned TWS, P, T, valid mask, and training-climatology anomalies for teammates.
- `data/processed/p_t_tws_baseline_predictions.nc`: every baseline's gridded prediction plus `test_eval_mask` and `spatial_fold`. Score deep-learning predictions on exactly these cells with `p_t_tws_baselines.evaluate_predictions`.
- `data/processed/p_t_tws_trend_maps.nc`, `data/processed/p_t_tws_correlation_maps.nc`: per-pixel trend and correlation maps.
- `results/metrics/p_t_tws_baseline_metrics_gracefo.csv`, `results/metrics/p_t_tws_baseline_metrics_spatial_cv.csv`, `results/metrics/p_t_tws_correlation_summary.csv`, `results/metrics/p_t_tws_trend_summary.csv`.
- `results/figures/p_t_tws_*.png`: driver-analysis and baseline figures.
- `models/p_t_tws_baselines.json`: feature sets, selected ridge strengths, gradient-boosting iterations, splits, and seed (fitted per-pixel weights are not stored).

Run all unit tests from the repository root with `python -m unittest discover -s tests`.

---

## Statistical Trend Analysis and Autoregressive Baseline

[notebooks/Statistical_Trend_Analysis.ipynb](notebooks/Statistical_Trend_Analysis.ipynb) runs on Google Colab (no GPU needed). It downloads the GRACE/GRACE-FO JPL mascon CRI file with `earthaccess`, or reuses it if already present, and prompts for Earthdata credentials at runtime.

- Trends: deseasonalized per-pixel Mann–Kendall with Hamed–Rao correction (first 3 lags), Sen's slope, and Benjamini–Hochberg FDR, for 2019–2025 and the full 2002–2025 record. Land cells use the in-file `land_mask`, |lat| ≤ 60°, with cos(lat) area weights.
- AR baseline: per-pixel AR(p), p ≤ 3 chosen by AIC, one-month-ahead. AIC needs no validation set, so the model is fit on train + validation (2019–2023) and scored on test (2024–2025), the same test months as the other baselines. It is compared against persistence and climatology using MAE, RMSE, correlation, and anomaly correlation. RMSE is computed per cell, then area-weighted.

Outputs:

- `results/metrics/stat_trend_*.csv`: significance summary, regional trends, AR metrics (global and regional), and a false-positive check on synthetic series.
- `results/figures/stat_trend_*.png`: trend maps for both windows, short- vs long-record trends, regional trends, significance by method, and the AR skill map.
- `data/processed/stat_trend_outputs.nc` (compressed, 4 MB): gridded Sen's slopes with confidence bounds, Mann–Kendall tau/p/q and significance for both windows, AR order, AR test predictions (2024–2025), and the deseasonalized TWS anomalies (2019–2025), on the |lat| ≤ 60° 0.5° grid. The notebook writes it uncompressed as `outputs/trend_ar_outputs.nc` (88 MB).

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
