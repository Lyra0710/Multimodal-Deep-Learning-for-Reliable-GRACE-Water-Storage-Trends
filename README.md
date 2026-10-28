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
models/                          saved baseline specifications
notebooks/                       interactive GRACE and trend analyses
results/figures/                 saved figures
results/metrics/                 saved metric tables
scripts/
  grace_fo_processing.py         GRACE-FO download and preprocessing
  grace_fo_persistence.py        one-month persistence baseline
  p_t_tws_data.py                GRACE + precipitation + temperature alignment
  p_t_tws_analysis.py            driver and trend analysis
  p_t_tws_baselines.py           regression and spatial-CV baselines
tests/                           root workflow tests
```

## Work completed

- GRACE-FO preprocessing, EDA, chronological split files, and a one-month persistence baseline.
- Statistical trend analysis and an autoregressive baseline in [Statistical_Trend_Analysis.ipynb](notebooks/Statistical_Trend_Analysis.ipynb).
- Precipitation/temperature alignment, driver analysis, and temporal and geographic regression baselines.
- Combined-data exploratory analysis and a per-cell Ridge baseline in [Combined_Dataset_Analysis_Ridge.ipynb](contributions/combined_dataset_analysis_ridge/Combined_Dataset_Analysis_Ridge.ipynb).

Saved figures and summaries are in `results/figures/` and `results/metrics/`; the combined Ridge contribution also contains its own CSV summaries and figures. See the current implementations: [GRACE preprocessing](scripts/grace_fo_processing.py), [persistence](scripts/grace_fo_persistence.py), [P/T alignment](scripts/p_t_tws_data.py), [P/T analysis](scripts/p_t_tws_analysis.py), [P/T baselines](scripts/p_t_tws_baselines.py), and [combined Ridge](contributions/combined_dataset_analysis_ridge/scripts/combined_ridge.py).

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

   Open [GRACE_GRACE_FO_Data_Analysis.ipynb](notebooks/GRACE_GRACE_FO_Data_Analysis.ipynb), [P_T_TWS_Drivers_and_Baselines.ipynb](notebooks/P_T_TWS_Drivers_and_Baselines.ipynb), or [Statistical_Trend_Analysis.ipynb](notebooks/Statistical_Trend_Analysis.ipynb) for the corresponding interactive analyses. The notebooks require their source data and, where applicable, Earthdata credentials; run their cells manually in order.

   Root tests can be run with:

   ```bash
   python -m unittest discover -s tests
   ```

## Evaluation setup

The main GRACE-FO persistence workflow forecasts one target month from the prior month. Its configured splits are train (2019–2022), validation (2023), and test (2024–2025). Land evaluation is limited to 60°S–60°N. Its primary RMSE is temporal RMSE computed per grid cell and then averaged using cosine-latitude weights.

The precipitation/temperature temporal baselines read the same split configuration, but their exact valid grid cells depend on aligned-data availability and the shared finite-prediction mask. The combined Ridge contribution uses the same nominal split boundaries, but its saved run covers January 2024–September 2025 because the supplied combined input ended in September; it reports both pooled and per-cell aggregation tables. Do not compare scores across workflows without matching dates, valid cells, and RMSE aggregation.

## Next steps

- Implement LSTM, TCN, and temporal Transformer models.
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

The project proposal/intermediate report is not present in this repository, so no repository link can be provided for it.
