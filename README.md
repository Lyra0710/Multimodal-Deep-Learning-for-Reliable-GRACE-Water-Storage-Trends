# Multimodal Deep Learning for Reliable GRACE Water Storage Trends

DATA 255 Group 5 project.

## Project overview

This project studies whether terrestrial water-storage anomalies (TWSA) from NASA GRACE and GRACE-FO can be predicted reliably from their recent history and environmental conditions. The prediction target is GRACE/GRACE-FO liquid-water-equivalent thickness anomaly; precipitation and near-surface temperature are the environmental predictors.

The current stage emphasizes data preparation, exploratory and trend analysis, and transparent baseline models. Deep-learning models and multimodal fusion are planned for the next stage.

## Datasets

- **GRACE/GRACE-FO:** JPL monthly mascon equivalent-water-height anomalies. The processing script uses `TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4` (RL06.3Mv04), including `lwe_thickness` in centimeters. [Dataset page](https://podaac.jpl.nasa.gov/dataset/TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4).
- **GPM IMERG:** monthly precipitation, configured as `GPM_3IMERGM`, version 07. [GES DISC dataset search](https://disc.gsfc.nasa.gov/datasets?keywords=GPM_3IMERGM_07).
- **MERRA-2:** monthly mean 2 m air temperature, configured as `M2TMNXSLV`, version 5.12.4. [GES DISC dataset search](https://disc.gsfc.nasa.gov/datasets?keywords=M2TMNXSLV_5.12.4).

Original downloads belong in `data/raw/` (with optional `imerg/` and `merra2/` subdirectories). Generated NetCDF files belong in `data/processed/`. Large source data and the combined aligned dataset are not included in this repository.

## Repository structure

```text
configs/                         shared split and P/T settings
contributions/combined_dataset_analysis_ridge/
  Combined_Dataset_Analysis_Ridge.ipynb
  scripts/combined_ridge.py      combined-data Ridge workflow
data/raw/                        local NASA source files (not committed)
data/processed/                  generated analysis/model inputs
models/                          saved baseline specifications; LSTM weights in models/lstm/
notebooks/                       interactive GRACE and trend analyses
results/figures/                 saved figures
results/metrics/                 saved metric tables
scripts/
  grace_fo_processing.py         GRACE-FO download and preprocessing
  grace_fo_persistence.py        one-month persistence baseline
  p_t_tws_data.py                GRACE + precipitation + temperature alignment
  p_t_tws_analysis.py            driver and trend analysis
  p_t_tws_baselines.py           regression and spatial-CV baselines
  lstm_forecast.py               LSTM ensemble forecaster with prediction intervals
tests/                           root workflow tests
```

## Work completed

- GRACE-FO preprocessing, EDA, chronological split files, and a one-month persistence baseline.
- Statistical trend analysis and an autoregressive baseline in [Statistical_Trend_Analysis.ipynb](notebooks/Statistical_Trend_Analysis.ipynb).
- Precipitation/temperature alignment, driver analysis, and temporal and geographic regression baselines.
- Combined-data exploratory analysis and a per-cell Ridge baseline in [Combined_Dataset_Analysis_Ridge.ipynb](contributions/combined_dataset_analysis_ridge/Combined_Dataset_Analysis_Ridge.ipynb).
- GRACE-only LSTM ensemble with calibrated prediction intervals in [lstm_forecast.py](scripts/lstm_forecast.py).

Saved figures and summaries are in `results/figures/` and `results/metrics/`; the combined Ridge contribution also contains its own CSV summaries and figures. See the current implementations: [GRACE preprocessing](scripts/grace_fo_processing.py), [persistence](scripts/grace_fo_persistence.py), [P/T alignment](scripts/p_t_tws_data.py), [P/T analysis](scripts/p_t_tws_analysis.py), [P/T baselines](scripts/p_t_tws_baselines.py), and [combined Ridge](contributions/combined_dataset_analysis_ridge/scripts/combined_ridge.py).

## Shared data handoff

The GRACE-FO master is `data/processed/grace_fo_processed.nc`; it contains monthly `lwe_thickness` (cm), `uncertainty`, `land_mask`, `mascon_ID`, `scale_factor`, and `GAD` on `time × lat × lon`. The chronological target-month handoff files are `grace_fo_train.nc` (2019–2022), `grace_fo_validation.nc` (2023), and `grace_fo_test.nc` (2024–2025). Use the master—not independently opened split files—when an input window crosses a split boundary.

The multimodal handoff produced by `p_t_tws_data.py` is `data/processed/p_t_tws_aligned_0.5deg.nc`. It contains `tws` (cm), `precip` (mm/month), `t2m` (degC), training-climatology anomaly versions of those fields, and two static masks. `mask` preserves the existing full-record baseline eligibility; `training_mask` uses only the configured training period and is the appropriate static cell-selection mask for leakage-free model development. Per-month finite-value checks are still required when constructing each model's inputs and targets. The aligned file is intentionally untracked because of its size; obtain the exact shared artifact or reproduce it before beginning multimodal development.

## How to reproduce the results

1. Use Python 3.10+ and work from the repository root. Create and activate an environment, then install the packages used by the root workflows:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install numpy pandas xarray matplotlib PyYAML netCDF4 python-dotenv podaac-data-subscriber earthaccess scipy scikit-learn h5netcdf
   ```

   The combined Ridge package has its own pinned dependency ranges in [requirements.txt](contributions/combined_dataset_analysis_ridge/requirements.txt). Install those as well when reproducing that workflow:

   ```bash
   python -m pip install -r contributions/combined_dataset_analysis_ridge/requirements.txt
   ```

2. To download data, create `.env` from the checked-in template and provide NASA Earthdata credentials. For IMERG and MERRA-2 access, authorize the **NASA GESDISC DATA ARCHIVE** application in the Earthdata profile.

   ```bash
   cp .env.example .env
   ```

   You may keep the repository in local storage or in Google Drive. If your data live elsewhere, either pass that location through the commands below where an argument is available, or update the `data.raw_dir` and `data.cache_dir` paths in [p_t_tws_config.yaml](configs/p_t_tws_config.yaml). Keep the GRACE source, cache, and outputs on storage with sufficient space.

3. Build the 2019–2025 GRACE-FO master file and optional land-only view. This command downloads the source when it is not already in `data/raw/`:

   ```bash
   python scripts/grace_fo_processing.py \
     --download-dir data/raw \
     --output data/processed/grace_fo_processed.nc \
     --land-view-output data/processed/grace_fo_land_modeling_view.nc \
     --env-file .env
   ```

   If the original NetCDF is already stored locally, replace `--download-dir data/raw` with `--input /path/to/original.nc`.

4. Create the GRACE-FO chronological handoff files and persistence results:

   ```bash
   python scripts/grace_fo_persistence.py \
     --input data/processed/grace_fo_processed.nc \
     --config configs/grace_fo_config.yaml \
     --metrics-dir results/metrics \
     --figures-dir results/figures \
     --split-output-dir data/processed \
     --model-output models/grace_fo_one_month_persistence_baseline.json
   ```

5. Build the aligned GRACE/GRACE-FO, IMERG, and MERRA-2 file, then run driver analysis and baselines. The first command reuses a `GRCTellus*CRI*.nc` file in `data/raw/` or downloads it; it streams IMERG/MERRA-2 by default and caches reduced driver data under `data/processed/p_t_tws_cache/`.

   ```bash
   python scripts/p_t_tws_data.py --config configs/p_t_tws_config.yaml --output data/processed/p_t_tws_aligned_0.5deg.nc --env-file .env

   python scripts/p_t_tws_analysis.py --input data/processed/p_t_tws_aligned_0.5deg.nc --config configs/p_t_tws_config.yaml --metrics-dir results/metrics --figures-dir results/figures --maps-dir data/processed

   python scripts/p_t_tws_baselines.py --input data/processed/p_t_tws_aligned_0.5deg.nc --config configs/p_t_tws_config.yaml --metrics-dir results/metrics --figures-dir results/figures --predictions-output data/processed/p_t_tws_baseline_predictions.nc --model-output models/p_t_tws_baselines.json
   ```

6. For the combined Ridge workflow, use the aligned file created in step 5. It is required input and is not tracked in Git. The command writes a separate rerun directory:

   ```bash
   python contributions/combined_dataset_analysis_ridge/scripts/combined_ridge.py \
     --input data/processed/p_t_tws_aligned_0.5deg.nc \
     --config configs/grace_fo_config.yaml \
     --output-dir contributions/combined_dataset_analysis_ridge/rerun_results
   ```

7. Train and evaluate the LSTM ensemble (needs PyTorch: `python -m pip install torch`). The first command reads the step-4 split files; the other two read the step-5 aligned file for the shared model comparison. Each takes about 4 minutes on a GPU. The reported results come from a Google Colab T4 GPU run, saved with its outputs in [LSTM_Colab_Run.ipynb](notebooks/LSTM_Colab_Run.ipynb):

   ```bash
   python scripts/lstm_forecast.py --config configs/lstm_config.yaml
   python scripts/lstm_forecast.py --config configs/lstm_aligned_grace.yaml
   python scripts/lstm_forecast.py --config configs/lstm_aligned_combined.yaml
   ```

   [LSTM_Forecast_and_Uncertainty.ipynb](notebooks/LSTM_Forecast_and_Uncertainty.ipynb) runs the same script and shows the results; it also runs on Google Colab (GPU optional).

   Open [GRACE_GRACE_FO_Data_Analysis.ipynb](notebooks/GRACE_GRACE_FO_Data_Analysis.ipynb), [P_T_TWS_Drivers_and_Baselines.ipynb](notebooks/P_T_TWS_Drivers_and_Baselines.ipynb), or [Statistical_Trend_Analysis.ipynb](notebooks/Statistical_Trend_Analysis.ipynb) for the corresponding interactive analyses. The notebooks require their source data and, where applicable, Earthdata credentials; run their cells manually in order.

   Root tests can be run with:

   ```bash
   python -m unittest discover -s tests
   ```

## Evaluation setup

The main GRACE-FO persistence workflow forecasts one target month from the prior month. Its configured splits are train (2019–2022), validation (2023), and test (2024–2025). Land evaluation is limited to 60°S–60°N. Its primary RMSE is temporal RMSE computed per grid cell and then averaged using cosine-latitude weights.

The precipitation/temperature temporal baselines read the same split configuration, but their exact valid grid cells depend on aligned-data availability and the shared finite-prediction mask. The combined Ridge contribution uses the same nominal split boundaries, but its saved run covers January 2024–September 2025 because the supplied combined input ended in September; it reports both pooled and per-cell aggregation tables. Do not compare scores across workflows without matching dates, valid cells, and RMSE aggregation.

## LSTM forecaster with prediction intervals

One LSTM is shared by all 46,325 land cells. For each cell and target month it reads the previous 12 month-to-month GRACE-FO TWSA changes (scaled per cell) and the calendar month, and predicts a Gaussian mean and variance for the next change; the forecast is TWSA(t − 1) plus that change. Inputs are GRACE-FO only. Five members with different seeds form a deep ensemble: the mean of member variances is data (aleatoric) uncertainty and the variance of member means is model (epistemic) uncertainty. Interval widths are rescaled by split-conformal calibration on 2023. Every input month precedes the target, per-cell scales and the baseline climatology use training months only, settings and early stopping use the validation year only, and the test years are scored once. Settings are in [lstm_config.yaml](configs/lstm_config.yaml).

Test results use the persistence workflow's months (2024-01 to 2025-12), cells and metrics. The AR baseline here is refitted on 2019–2022 only.

| Model | MAE (cm) | RMSE (cm) | Pearson r |
|---|---|---|---|
| LSTM, 5-member ensemble | **2.262** | **2.871** | **0.826** |
| Seasonal persistence | 2.435 | 3.066 | 0.807 |
| AR (p ≤ 3, anomalies) | 2.605 | 3.250 | 0.804 |
| Persistence | 3.437 | 4.201 | 0.764 |
| Climatology | 6.642 | 7.440 | 0.601 |

The LSTM has lower RMSE than persistence over 89% of the land area and lower than AR over 73%; it is weaker than seasonal persistence in the Amazon and East Africa. RMSE is 2.93 cm in 2024 and 2.72 cm in 2025. Calibrated 50/80/90/95% intervals cover 49.3/80.1/90.0/94.9% of test values (mean 90% width 10.1 cm). Predicted σ tracks the actual error (Spearman ρ = 0.59), and about 99% of the predicted variance is aleatoric. Outputs are `results/metrics/lstm_*`, `results/figures/lstm_*` and `models/lstm/`.

### LSTM runs on the aligned file (shared comparison)

Two further runs use the aligned GRACE/IMERG/MERRA-2 file from step 5, its cell mask, months from 2018 as input history, and the test period 2024-01 to 2025-09 (the file ends in September 2025). [lstm_aligned_grace.yaml](configs/lstm_aligned_grace.yaml) uses GRACE TWS only. [lstm_aligned_combined.yaml](configs/lstm_aligned_combined.yaml) adds past precipitation and temperature anomalies to each monthly input (early fusion). Both are strict forecasts: every input comes from the 12 months before the target. Model and training settings are unchanged. Test predictions with calibrated 90% bounds are saved in `results/predictions/` on the evaluator grid (`time × lat × lon`, cm).

Scores on the aligned file's 46,325 cells, 21 test months:

| Model | MAE (cm) | RMSE (cm) | Pearson r | 90% interval coverage |
|---|---|---|---|---|
| LSTM, GRACE + precipitation + temperature | **2.266** | **2.858** | **0.820** | 89.4% |
| LSTM, GRACE only | 2.286 | 2.899 | 0.809 | 89.6% |
| Seasonal persistence | 2.453 | 3.083 | 0.788 | |
| AR (p ≤ 3, anomalies) | 2.620 | 3.261 | 0.787 | |
| Persistence | 3.482 | 4.249 | 0.742 | |

Adding precipitation and temperature lowers the LSTM's per-cell RMSE by 0.041 cm (1.4%) and narrows the mean 90% interval from 10.0 to 9.5 cm at the same coverage.

The Transformer evaluator ([PR #1](https://github.com/Lyra0710/Multimodal-Deep-Learning-for-Reliable-GRACE-Water-Storage-Trends/pull/1)) scores models on the cell-months that every model and its reference file share. Its reference file has 46,497 cells, of which 45,956 are also in the aligned file used here; observed TWS is identical on those cells. Scored with that evaluator's metric functions on the 45,956 common cells and 21 months:

| Model | Per-cell RMSE (cm) | MAE (cm) | Pooled RMSE (cm) | Pearson r |
|---|---|---|---|---|
| LSTM, GRACE + precipitation + temperature | **2.856** | **2.264** | **3.552** | **0.820** |
| LSTM, GRACE only | 2.898 | 2.285 | 3.570 | 0.808 |
| Seasonal persistence | 2.950 | 2.342 | 3.577 | 0.797 |
| Transformer, GRACE + precipitation + temperature | 3.030 | 2.415 | 3.678 | 0.795 |
| Transformer, GRACE only | 3.180 | 2.535 | 3.910 | 0.780 |
| Ridge | 3.688 | 3.075 | 4.727 | 0.811 |
| Persistence | 4.249 | 3.481 | 5.784 | 0.742 |

The two reference masks differ by about 900 mostly coastal cells; the team should fix one aligned file before final scoring. To add the LSTM to the evaluator, append `--model lstm_grace ../../results/predictions/lstm_aligned_grace_test_predictions.nc prediction` and `--model lstm_combined ../../results/predictions/lstm_aligned_combined_test_predictions.nc prediction` to `compare.sh`.

## Next steps

- Implement TCN and temporal Transformer models.
- Investigate early and late multimodal fusion.
- Evaluate a gridded ConvLSTM model, as suggested in proposal feedback.
- Test geographic, temporal, and GRACE-to-GRACE-FO generalization.
- Evaluate prediction uncertainty and Integrated Gradients explanations.
- Verify leakage prevention and compare final models against the existing baselines.

## References

- [JPL GRACE/GRACE-FO Mascon CRI RL06.3Mv04](https://podaac.jpl.nasa.gov/dataset/TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4)
- [GPM IMERG at NASA GES DISC](https://disc.gsfc.nasa.gov/datasets?keywords=GPM_3IMERGM_07)
- [MERRA-2 at NASA GES DISC](https://disc.gsfc.nasa.gov/datasets?keywords=M2TMNXSLV_5.12.4)
- [Project repository](https://github.com/Lyra0710/Multimodal-Deep-Learning-for-Reliable-GRACE-Water-Storage-Trends)
